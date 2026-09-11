"""
CameraManager owns exactly one active CameraSource at a time and runs
the inference worker thread that consumes frames from it.

Thread-safety model:
- `self._lock` guards the small shared state /webcam/status,
  /webcam/start|stop and the detections WebSocket touch concurrently
  (source ref, stats, threshold, subscribers, the session/track state,
  and the MissionRecorder). Held only for quick work -- never across a
  GStreamer start()/stop() or a YOLO call.
- Start/stop of the pipeline is serialized by `self._lifecycle_lock`.

Transports that read from this class:
- Detection JSON, over /webcam/status (polled) and /webcam/ws/detections
  (pushed) -- driven by the inference worker's output. Every detection
  carries `bbox` (pixels in the inference frame) AND `bbox_norm`
  (0..1), plus the payload carries `frame_w`/`frame_h`. The browser
  overlays using `bbox_norm` so it does not matter that the WHEP video
  is a different size from the inference frame (Phase B).
- Raw video frames: streamed to MediaMTX by MediaMTXPublisher via its
  own consumer id on the same LatestFrameBuffer, in its own pipeline.

MissionRecorder (camera/mission.py) sits on top of the session state:
times the mission, logs events, evaluates alert rules once per frame.

Phase A: recording enabled -> an independent cv2.VideoWriter captures
the annotated frames to <MISSION_RECORD_DIR>/mission_<session_id>.mp4,
fully decoupled from the publisher, never able to stall inference.

Phase C: the mission's confidence threshold is set per-mission via
set_mission_config(confidence=...) from /webcam/session/config
(Settings -> defaultConfidence). set_confidence() stays as a manual
override.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import os
import threading
import time
import uuid
from collections import Counter, deque
from typing import Optional

import cv2

from .base import CameraError, CameraInfo, CameraSource
from .config import load_camera_config
from .csi_source import CSICameraSource
from .custom_source import CustomCameraSource
from .discovery import discover_all
from .frame_buffer import Frame
from .mission import MissionRecorder, format_offset
from .publisher import MediaMTXPublisher
from .rtsp_source import RTSPCameraSource
from .v4l2_source import V4L2CameraSource

logger = logging.getLogger("camera.manager")


MIN_TRACK_HITS = 3

_FPS_WINDOW_SAMPLES = 60
_LATENCY_WINDOW_SAMPLES = 30
_STALL_SECONDS = 2.0


# ------------------------------------------------------------------
# Mission recording (Phase A)
# ------------------------------------------------------------------
MISSION_RECORD_DIR = os.getenv(
    "MISSION_RECORD_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "recordings")),
)
os.makedirs(MISSION_RECORD_DIR, exist_ok=True)

_RECORD_FPS = float(os.getenv("MISSION_RECORD_FPS", "15"))
_RECORD_FOURCC = os.getenv("MISSION_RECORD_FOURCC", "mp4v")  # match video_detector.py
_RECORD_ANNOTATE = os.getenv("MISSION_RECORD_ANNOTATE", "1") == "1"
_RECORD_MIN_BYTES = 10_000


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else float(v)


def _norm_bbox(x1, y1, x2, y2, w, h):
    """Phase B: pixel bbox -> [x1,y1,x2,y2] in 0..1 against a w x h frame."""
    if not w or not h:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        round(_clamp01(x1 / w), 6),
        round(_clamp01(y1 / h), 6),
        round(_clamp01(x2 / w), 6),
        round(_clamp01(y2 / h), 6),
    ]


def _queue_put_latest(queue: "asyncio.Queue", payload: dict) -> None:
    try:
        queue.get_nowait()
    except asyncio.QueueEmpty:
        pass
    try:
        queue.put_nowait(payload)
    except asyncio.QueueFull:
        pass


class CameraManager:
    def __init__(self, detector, config_path: Optional[str] = None):
        self.detector = detector
        self._config_path = config_path
        self._lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()

        self._source: Optional[CameraSource] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._worker_running = False

        self._confidence_threshold = 0.25
        self._latest_detections: list = []
        self._last_error: Optional[str] = None

        # Phase B: geometry of the frame the model last ran on. The
        # detections' bbox_norm are normalized against this.
        self._last_frame_w = 0
        self._last_frame_h = 0

        # SESSION state -- keyed by ByteTrack track_id.
        self._session_id: Optional[str] = None
        self._session_source = "Webcam"
        self._session_started_at: Optional[str] = None
        self._session_start_monotonic = 0.0
        self._session_frame_count = 0
        self._session_detections_seen = 0
        self._tracks: dict[int, dict] = {}
        self._display_map: dict[int, int] = {}

        # Measured statistics only.
        self._inference_fps = 0.0
        self._inference_time_ms = 0.0
        self._processing_latency_ms = 0.0
        self._frame_completion_times: "deque[float]" = deque(maxlen=_FPS_WINDOW_SAMPLES)
        self._latency_samples: "deque[float]" = deque(maxlen=_LATENCY_WINDOW_SAMPLES)
        self._last_frame_monotonic = 0.0

        # Mission layer + pending config for the NEXT start().
        self._mission = MissionRecorder()
        self._pending_alert_rules: list = []
        self._pending_recording = False
        self._pending_confidence: Optional[float] = None  # Phase C

        # --- mission recording (Phase A) ---
        self._recording_enabled = False
        self._record_writer = None
        self._record_path: Optional[str] = None
        self._record_rel_url: Optional[str] = None
        self._record_frames_written = 0
        self._record_started_monotonic = 0.0

        self._loop: Optional["asyncio.AbstractEventLoop"] = None
        self._detection_subscribers: "set[asyncio.Queue]" = set()

        self._publisher = MediaMTXPublisher(
            rtsp_url=os.getenv("MEDIAMTX_RTSP_URL", "rtsp://127.0.0.1:8554/cam"),
            width=int(os.getenv("WEBCAM_PUBLISH_WIDTH", "960")),
            height=int(os.getenv("WEBCAM_PUBLISH_HEIGHT", "540")),
            fps=int(os.getenv("WEBCAM_PUBLISH_FPS", "25")),
            bitrate_bps=int(os.getenv("WEBCAM_PUBLISH_BITRATE", "2500000")),
        )

    # ------------------------------------------------------------------
    def set_mission_config(
        self,
        *,
        alert_rules: Optional[list],
        recording: bool,
        confidence: Optional[float] = None,
    ) -> None:
        with self._lock:
            self._pending_alert_rules = list(alert_rules or [])
            self._pending_recording = bool(recording)
            self._pending_confidence = (
                float(confidence) if confidence is not None else None
            )

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def discover_cameras(self) -> list[dict]:
        try:
            config = load_camera_config(self._config_path) if self._config_path else load_camera_config()
            infos = discover_all(config)
            return [info.public_dict() for info in infos]
        except Exception:
            logger.exception("Camera discovery failed")
            return []

    def _resolve_camera_info(self, camera_id: str) -> CameraInfo:
        config = load_camera_config(self._config_path) if self._config_path else load_camera_config()
        for info in discover_all(config):
            if info.id == camera_id:
                return info
        raise CameraError(f"Unknown camera_id '{camera_id}'. Call /webcam/cameras first.")

    def _build_source(self, info: CameraInfo, config) -> CameraSource:
        if info.type == "USB" or info.type == "HDMI":
            usb_cfg = next((c for c in config.usb_cameras if c.get("device") == info.device), {})
            return V4L2CameraSource(
                info,
                device=info.device,
                width=usb_cfg.get("width", 1280),
                height=usb_cfg.get("height", 720),
                fps=usb_cfg.get("fps", 30),
                use_mjpeg=usb_cfg.get("use_mjpeg", True),
            )
        if info.type == "CSI":
            csi_cfg = next((c for c in config.csi_cameras if c["id"] == info.id), {})
            return CSICameraSource(
                info,
                sensor_id=info.sensor_id or 0,
                width=csi_cfg.get("width", 1280),
                height=csi_cfg.get("height", 720),
                fps=csi_cfg.get("fps", 30),
            )
        if info.type == "RTSP":
            rtsp_cfg = next((c for c in config.rtsp_cameras if c["id"] == info.id), {})
            return RTSPCameraSource(
                info,
                url=rtsp_cfg["url"],
                latency_ms=rtsp_cfg.get("rtsp_latency_ms", 100),
                protocol=rtsp_cfg.get("rtsp_protocol", "tcp"),
                codec=rtsp_cfg.get("codec", "h264"),
            )
        if info.type == "CUSTOM":
            custom_cfg = next((c for c in config.custom_cameras if c["id"] == info.id), {})
            return CustomCameraSource(info, pipeline=custom_cfg["pipeline"])
        raise CameraError(f"Unsupported camera type '{info.type}'")

    # ------------------------------------------------------------------
    # Start / stop
    # ------------------------------------------------------------------
    def start(self, camera_id: str) -> dict:
        with self._lifecycle_lock:
            self.stop()  # idempotent

            config = load_camera_config(self._config_path) if self._config_path else load_camera_config()
            info = self._resolve_camera_info(camera_id)
            source = self._build_source(info, config)

            try:
                source.start()
            except CameraError:
                raise
            except Exception as exc:
                raise CameraError(f"Failed to start camera '{camera_id}': {exc}") from exc

            self.detector.reset_tracker()

            resolution = getattr(source, "resolution", None) or info.resolution

            with self._lock:
                self._source = source
                self._latest_detections = []
                self._last_error = None
                self._last_frame_w = 0
                self._last_frame_h = 0
                self._inference_fps = 0.0
                self._inference_time_ms = 0.0
                self._processing_latency_ms = 0.0
                self._frame_completion_times.clear()
                self._latency_samples.clear()
                self._last_frame_monotonic = 0.0

                self._session_id = uuid.uuid4().hex[:12]
                self._session_source = "Webcam"
                self._session_started_at = datetime.datetime.utcnow().isoformat() + "Z"
                self._session_start_monotonic = time.monotonic()
                self._session_frame_count = 0
                self._session_detections_seen = 0
                self._tracks = {}
                self._display_map = {}

                # Phase C: apply the mission confidence BEFORE begin() so
                # the recorder logs the value that was actually used.
                if self._pending_confidence is not None:
                    self._confidence_threshold = self._pending_confidence

                # NEW MISSION.
                self._mission.begin(
                    session_id=self._session_id,
                    source_type=info.type or "Webcam",
                    source_name=info.name or info.type or "Webcam",
                    resolution=resolution,
                    recording_enabled=self._pending_recording,
                    alert_rules=self._pending_alert_rules,
                    confidence=self._confidence_threshold,
                )

                # Phase A: arm recording BEFORE the pending flag is consumed.
                self._recording_enabled = bool(self._pending_recording)
                self._record_writer = None
                self._record_path = None
                self._record_rel_url = None
                self._record_frames_written = 0
                self._record_started_monotonic = 0.0

                self._pending_alert_rules = []
                self._pending_recording = False
                self._pending_confidence = None

            self._worker_running = True
            self._worker_thread = threading.Thread(target=self._inference_loop, daemon=True)
            self._worker_thread.start()

            self._publisher.start(self._read_publish_frame)

            return {
                "camera_id": info.id,
                "type": info.type,
                "device": info.device,
                "resolution": resolution,
            }

    def stop(self) -> None:
        self._publisher.stop()

        self._worker_running = False
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=3)
            self._worker_thread = None

        # Worker has stopped -> no writes can race the release.
        record_url = self._finalize_recording()

        with self._lock:
            source = self._source
            self._source = None
            self._latest_detections = []
            if record_url:
                self._mission.set_video(record_url)
            self._mission.end()

        if source is not None:
            try:
                source.stop()
            except Exception:
                logger.exception("Error while stopping camera source")

    def clear_session(self) -> None:
        with self._lock:
            self._session_id = None
            self._session_started_at = None
            self._session_start_monotonic = 0.0
            self._session_frame_count = 0
            self._session_detections_seen = 0
            self._tracks = {}
            self._display_map = {}
            self._last_frame_w = 0
            self._last_frame_h = 0
            self._mission.clear()
            self._recording_enabled = False
            self._safe_close_writer()
            self._record_path = None
            self._record_rel_url = None
            self._record_frames_written = 0

    # ------------------------------------------------------------------
    # Inference worker
    # ------------------------------------------------------------------
    def _inference_loop(self) -> None:
        source = self._source
        if source is None:
            return

        while self._worker_running and source.is_running:
            frame_obj = source.get_latest_frame(timeout=1.0)
            if frame_obj is None:
                if source.last_error:
                    with self._lock:
                        self._last_error = source.last_error
                continue

            frame = frame_obj.image
            frame_h, frame_w = frame.shape[:2]  # Phase B
            inference_start = time.perf_counter()
            frame_age_ms = max(0.0, (time.monotonic() - frame_obj.timestamp) * 1000.0)

            with self._lock:
                threshold = self._confidence_threshold
                display_map = dict(self._display_map)
            tracker_feed_conf = self.detector.tracker_feed_conf

            try:
                result = self.detector.track_frame(
                    frame, conf=min(tracker_feed_conf, threshold)
                )
                error = None
            except Exception as exc:
                logger.exception("Inference error")
                result = {"detections": []}
                error = str(exc)

            inference_ms = (time.perf_counter() - inference_start) * 1000.0
            processing_latency_ms = frame_age_ms + inference_ms

            detections = []
            for det in result.get("detections", []):
                confidence = float(det.get("confidence", 0))
                if confidence < threshold:
                    continue
                bbox = det.get("bbox", [])
                if len(bbox) != 4:
                    continue
                tid = det.get("track_id")
                bx1, by1, bx2, by2 = (float(v) for v in bbox)
                detections.append(
                    {
                        "track_id": tid,
                        "display_id": display_map.get(tid),
                        "class": det.get("class", "unknown"),
                        "class_id": det.get("class_id", -1),
                        "confidence": round(confidence, 4),
                        "bbox": [int(bx1), int(by1), int(bx2), int(by2)],
                        # Phase B: resolution-independent box.
                        "bbox_norm": _norm_bbox(bx1, by1, bx2, by2, frame_w, frame_h),
                    }
                )

            now_perf = time.perf_counter()
            now_mono = time.monotonic()
            with self._lock:
                self._latest_detections = detections
                self._last_frame_w = frame_w
                self._last_frame_h = frame_h
                self._inference_time_ms = inference_ms
                self._last_error = error
                self._last_frame_monotonic = now_mono

                self._frame_completion_times.append(now_perf)
                if len(self._frame_completion_times) >= 2:
                    span = self._frame_completion_times[-1] - self._frame_completion_times[0]
                    if span > 0:
                        self._inference_fps = (len(self._frame_completion_times) - 1) / span

                self._latency_samples.append(processing_latency_ms)
                self._processing_latency_ms = (
                    sum(self._latency_samples) / len(self._latency_samples)
                )

                resolution = getattr(source, "resolution", None)
                self._mission.set_resolution(resolution)

                self._session_frame_count += 1
                self._session_detections_seen += len(detections)
                mission_offset = self._mission.offset_now

                for det in detections:
                    track_id = det.get("track_id")
                    if track_id is None:
                        continue
                    bbox = det["bbox"]
                    conf = det["confidence"]
                    entry = self._tracks.get(track_id)
                    if entry is None:
                        entry = {
                            "classes": Counter(),
                            "confidences": [],
                            "first_seen_frame": self._session_frame_count,
                            "last_seen_frame": self._session_frame_count,
                            "first_seen_offset": mission_offset,
                            "last_seen_offset": mission_offset,
                            "first_bbox": list(bbox),
                            "last_bbox": list(bbox),
                            "best_bbox": list(bbox),
                            "best_confidence": conf,
                        }
                        self._tracks[track_id] = entry
                    entry["classes"][det["class"]] += 1
                    entry["confidences"].append(conf)
                    entry["last_seen_frame"] = self._session_frame_count
                    entry["last_seen_offset"] = mission_offset
                    entry["last_bbox"] = list(bbox)
                    if conf > entry["best_confidence"]:
                        entry["best_confidence"] = conf
                        entry["best_bbox"] = list(bbox)

                visible_class_counts: dict = {}
                for det in detections:
                    visible_class_counts[det["class"]] = visible_class_counts.get(det["class"], 0) + 1

                unique_class_counts: dict = {}
                for entry in self._tracks.values():
                    if len(entry["confidences"]) < MIN_TRACK_HITS or not entry["classes"]:
                        continue
                    cls = entry["classes"].most_common(1)[0][0]
                    unique_class_counts[cls] = unique_class_counts.get(cls, 0) + 1

                self._mission.observe(
                    visible_class_counts=visible_class_counts,
                    unique_class_counts=unique_class_counts,
                )

            self._publish_detections(
                detections=detections,
                frame_id=frame_obj.frame_id,
                frame_timestamp=frame_obj.timestamp,
                resolution=resolution,
                frame_w=frame_w,
                frame_h=frame_h,
            )

            # Phase A -- evidence recording (self-contained, never raises)
            self._record_tick(frame, detections)

    # ------------------------------------------------------------------
    # Detections WebSocket fan-out
    # ------------------------------------------------------------------
    def set_event_loop(self, loop: "asyncio.AbstractEventLoop") -> None:
        self._loop = loop

    def subscribe_detections(self) -> "asyncio.Queue":
        queue: "asyncio.Queue" = asyncio.Queue(maxsize=1)
        with self._lock:
            self._detection_subscribers.add(queue)
        return queue

    def unsubscribe_detections(self, queue: "asyncio.Queue") -> None:
        with self._lock:
            self._detection_subscribers.discard(queue)

    def _publish_detections(
        self, detections, frame_id, frame_timestamp, resolution, frame_w=None, frame_h=None
    ) -> None:
        with self._lock:
            subscribers = list(self._detection_subscribers)
            loop = self._loop
        if loop is None or not subscribers:
            return
        payload = {
            "type": "detections",
            "frame_id": frame_id,
            "timestamp": frame_timestamp,
            "resolution": resolution,
            # Phase B: the frame bbox_norm is normalized against.
            "frame_w": frame_w,
            "frame_h": frame_h,
            "detections": detections,
        }
        for queue in subscribers:
            loop.call_soon_threadsafe(_queue_put_latest, queue, payload)

    # ------------------------------------------------------------------
    # Raw-frame access
    # ------------------------------------------------------------------
    def _read_publish_frame(self):
        with self._lock:
            source = self._source
        if source is None:
            return None
        frame_obj = source.buffer.get_latest(
            block=True, timeout=1.0, consumer_id="mediamtx-publisher"
        )
        return frame_obj.image if frame_obj is not None else None

    def get_latest_raw_frame(self, consumer_id: str) -> Optional[Frame]:
        with self._lock:
            source = self._source
        if source is None:
            return None
        return source.buffer.get_latest(block=False, consumer_id=consumer_id)

    def forget_video_consumer(self, consumer_id: str) -> None:
        with self._lock:
            source = self._source
        if source is not None:
            source.buffer.forget_consumer(consumer_id)

    # ------------------------------------------------------------------
    # Mission recording (Phase A)
    # ------------------------------------------------------------------
    @staticmethod
    def _draw_record_boxes(img, detections) -> None:
        for det in detections:
            bbox = det.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = (int(v) for v in bbox)
            color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            did = det.get("display_id")
            label = (f"#{did} " if did else "") + str(det.get("class", "?"))
            conf = det.get("confidence")
            if isinstance(conf, (int, float)):
                label += f" {conf * 100:.0f}%"
            cv2.putText(
                img, label, (x1, max(14, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )

    def _safe_close_writer(self) -> None:
        writer = self._record_writer
        self._record_writer = None
        if writer is not None:
            try:
                writer.release()
            except Exception:
                logger.exception("VideoWriter.release() failed")

    def _record_tick(self, frame, detections) -> None:
        if not self._recording_enabled:
            return
        try:
            if self._record_writer is None:
                h, w = frame.shape[:2]
                sid = self._session_id or datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
                fname = f"mission_{sid}.mp4"
                self._record_path = os.path.join(MISSION_RECORD_DIR, fname)
                self._record_rel_url = f"/webcam/recording/{fname}"
                fourcc = cv2.VideoWriter_fourcc(*_RECORD_FOURCC)
                self._record_writer = cv2.VideoWriter(
                    self._record_path, fourcc, _RECORD_FPS, (int(w), int(h))
                )
                if not self._record_writer.isOpened():
                    logger.error("mission VideoWriter failed to open: %s", self._record_path)
                    self._record_writer = None
                    self._recording_enabled = False
                    return
                self._record_started_monotonic = time.monotonic()
                self._record_frames_written = 0
                logger.info(
                    "mission recording -> %s @ %.0f fps %dx%d",
                    self._record_path, _RECORD_FPS, w, h,
                )

            target_dt = (1.0 / _RECORD_FPS) if _RECORD_FPS > 0 else (1.0 / 15.0)
            elapsed = time.monotonic() - self._record_started_monotonic
            target_count = int(elapsed / target_dt) + 1
            need = target_count - self._record_frames_written
            if need <= 0:
                return

            rec = frame
            if _RECORD_ANNOTATE and detections:
                rec = frame.copy()
                self._draw_record_boxes(rec, detections)

            for _ in range(min(need, 5)):  # cap catch-up so a stall can't spiral
                self._record_writer.write(rec)
                self._record_frames_written += 1

        except Exception:
            logger.exception("mission recording tick failed; disabling recording")
            self._recording_enabled = False
            self._safe_close_writer()

    def _finalize_recording(self) -> Optional[str]:
        if self._record_writer is None:
            return None
        path = self._record_path
        rel = self._record_rel_url
        self._safe_close_writer()
        try:
            ok = bool(path) and os.path.exists(path) and os.path.getsize(path) > _RECORD_MIN_BYTES
        except OSError:
            ok = False
        if ok:
            logger.info(
                "mission recording finalized: %s (%d frames)",
                path, self._record_frames_written,
            )
            return rel
        logger.warning("mission recording produced no usable file: %s", path)
        return None

    # ------------------------------------------------------------------
    # Session / mission summary (survives stop())
    # ------------------------------------------------------------------
    def _build_session_summary(self) -> dict:
        with self._lock:
            tracks_snapshot = {
                track_id: {
                    "classes": Counter(entry["classes"]),
                    "confidences": list(entry["confidences"]),
                    "first_seen_frame": entry["first_seen_frame"],
                    "last_seen_frame": entry["last_seen_frame"],
                    "first_seen_offset": entry.get("first_seen_offset", 0.0),
                    "last_seen_offset": entry.get("last_seen_offset", 0.0),
                    "first_bbox": list(entry.get("first_bbox", []) or []),
                    "last_bbox": list(entry.get("last_bbox", []) or []),
                    "best_bbox": list(entry.get("best_bbox", []) or []),
                    "best_confidence": entry.get("best_confidence", 0.0),
                }
                for track_id, entry in self._tracks.items()
            }
            session_id = self._session_id
            started_at = self._session_started_at
            active = self._source is not None and self._source.is_running
            inference_fps = self._inference_fps
            processing_latency_ms = self._processing_latency_ms
            frame_w = self._last_frame_w
            frame_h = self._last_frame_h
            mission_meta = self._mission.to_dict()

        confirmed = []
        for track_id, entry in tracks_snapshot.items():
            confidences = entry["confidences"]
            if len(confidences) < MIN_TRACK_HITS or not entry["classes"]:
                continue
            confirmed.append((track_id, entry))
        confirmed.sort(key=lambda kv: kv[0])

        new_display_map = {track_id: i + 1 for i, (track_id, _) in enumerate(confirmed)}
        with self._lock:
            self._display_map = new_display_map

        tracks = []
        class_counts: dict = {}
        for track_id, entry in confirmed:
            confidences = entry["confidences"]
            track_class = entry["classes"].most_common(1)[0][0]
            average_confidence = round(sum(confidences) / len(confidences), 4)
            max_confidence = round(max(confidences), 4)
            duration_s = max(0.0, entry["last_seen_offset"] - entry["first_seen_offset"])

            best_bbox = entry["best_bbox"]
            rep_bbox_norm = None
            if frame_w and frame_h and len(best_bbox) == 4:
                rep_bbox_norm = _norm_bbox(
                    best_bbox[0], best_bbox[1], best_bbox[2], best_bbox[3], frame_w, frame_h
                )

            tracks.append(
                {
                    "track_id": track_id,
                    "display_id": new_display_map[track_id],
                    "class": track_class,
                    "first_seen_frame": entry["first_seen_frame"],
                    "last_seen_frame": entry["last_seen_frame"],
                    "first_seen": format_offset(entry["first_seen_offset"]),
                    "last_seen": format_offset(entry["last_seen_offset"]),
                    "duration_seconds": round(duration_s, 1),
                    "frames_seen": len(confidences),
                    "average_confidence": average_confidence,
                    "max_confidence": max_confidence,
                    "representative_confidence": round(entry["best_confidence"], 4),
                    "representative_bbox": best_bbox,
                    "representative_bbox_norm": rep_bbox_norm,  # Phase B
                    "first_bbox": entry["first_bbox"],
                    "last_bbox": entry["last_bbox"],
                }
            )
            class_counts[track_class] = class_counts.get(track_class, 0) + 1

        if tracks:
            average_confidence = round(
                sum(t["average_confidence"] for t in tracks) / len(tracks), 4
            )
            max_confidence = round(max(t["max_confidence"] for t in tracks), 4)
        else:
            average_confidence = None
            max_confidence = None

        summary = {
            "session_id": session_id,
            "source": mission_meta["source_name"],
            "started_at": started_at,
            "active": active,
            "object_count": len(tracks),
            "tracks_created": len(tracks_snapshot),
            "class_counts": class_counts,
            "tracks": tracks,
            "average_confidence": average_confidence,
            "max_confidence": max_confidence,
            "mission_id": mission_meta["mission_id"],
            "start_time": mission_meta["start_time"],
            "end_time": mission_meta["end_time"],
            "duration_seconds": mission_meta["duration_seconds"],
            "duration_hms": mission_meta["duration_hms"],
            "source_type": mission_meta["source_type"],
            "source_name": mission_meta["source_name"],
            "resolution": mission_meta["resolution"],
            "confidence": mission_meta.get("confidence"),  # Phase C
            "frame_w": frame_w or None,   # Phase B
            "frame_h": frame_h or None,   # Phase B
            "recording": mission_meta["recording"],
            "performance": {
                "fps": round(inference_fps, 2),
                "latency_ms": round(processing_latency_ms, 2),
            },
            "total_unique_objects": len(tracks),
            "events": mission_meta["events"],
            "alerts": mission_meta["alerts"],
            "alert_count": mission_meta["alert_count"],
            "alert_rule_count": mission_meta["alert_rule_count"],
            "highest_severity": mission_meta["highest_severity"],
        }
        return summary

    # ------------------------------------------------------------------
    # Readers used by the FastAPI routes
    # ------------------------------------------------------------------
    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._source is not None and self._source.is_running

    def get_status(self) -> dict:
        with self._lock:
            source = self._source
            detections = list(self._latest_detections)
            running = source is not None and source.is_running
            camera_id = source.info.id if source else None
            camera_type = source.info.type if source else None
            resolution = getattr(source, "resolution", None) if source else None
            confidence = self._confidence_threshold
            frame_w = self._last_frame_w
            frame_h = self._last_frame_h
            inference_fps = self._inference_fps
            inference_ms = self._inference_time_ms
            processing_latency_ms = self._processing_latency_ms
            last_frame_monotonic = self._last_frame_monotonic
            error = self._last_error
            buffer_stats = source.buffer.stats if source else {"dropped_frames": 0, "total_frames": 0}

        stale = (
            not running
            or last_frame_monotonic <= 0.0
            or (time.monotonic() - last_frame_monotonic) > _STALL_SECONDS
        )
        if stale:
            inference_fps = 0.0
            processing_latency_ms = 0.0

        class_counts: dict = {}
        for det in detections:
            class_counts[det["class"]] = class_counts.get(det["class"], 0) + 1

        return {
            "success": True,
            "running": running,
            "camera_id": camera_id,
            "camera_type": camera_type,
            "fps": round(inference_fps, 2),
            "inference_time_ms": round(inference_ms, 2),
            "processing_latency_ms": round(processing_latency_ms, 2),
            "stream_fps": round(inference_fps, 2),
            "resolution": resolution,
            "frame_w": frame_w or None,   # Phase B
            "frame_h": frame_h or None,   # Phase B
            "confidence": confidence,     # Phase C
            "object_count": len(detections),
            "class_counts": class_counts,
            "detections": detections,
            "frames_dropped": buffer_stats["dropped_frames"],
            "frames_captured": buffer_stats["total_frames"],
            "error": error,
            "session": self._build_session_summary(),
        }

    def get_mission(self) -> dict:
        return self._build_session_summary()

    def set_confidence(self, value: float) -> None:
        with self._lock:
            self._confidence_threshold = value

    def get_confidence(self) -> float:
        with self._lock:
            return self._confidence_threshold

    def shutdown(self) -> None:
        self.stop()
