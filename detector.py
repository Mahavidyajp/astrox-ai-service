import os
import threading

import cv2
import numpy as np
from ultralytics import YOLO

# Resolved once, absolutely, so tracker="..." always finds THIS
# project's bytetrack.yaml regardless of the process's current working
# directory (uvicorn may be launched from a different cwd than this
# file lives in). Previously no bytetrack.yaml existed in the project
# at all, so Ultralytics silently fell back to its own packaged
# default config -- not reviewable, not tunable, not documented. See
# bytetrack.yaml in this same directory for the tuned values and the
# reasoning behind each one.
_TRACKER_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bytetrack.yaml")


def _read_tracker_feed_conf(path: str, default: float = 0.1) -> float:
    """
    Read `track_low_thresh` out of the ByteTrack config file.

    Tracking callers (CameraManager) feed THIS value -- a LOW floor --
    into model.track(), NOT the operator's display confidence
    threshold. Reason: ByteTrack associates in two stages. Stage one
    matches detections >= track_high_thresh to active tracks. Stage
    two uses the weaker detections in
    [track_low_thresh, track_high_thresh) purely to keep an
    ALREADY-active track alive through a brief confidence dip (motion
    blur, partial occlusion, pose/range change). If model.track() is
    given the operator threshold as its `conf`, every sub-threshold
    box is pruned before ByteTrack ever sees it, stage two is
    permanently starved, and a tracked object that dips for a few
    frames is dropped and then re-created with a NEW id on recovery
    (ID churn -> inflated unique counts, runaway track ids). Feeding
    the low floor here, and applying the operator threshold afterward
    to what is shown / counted, fixes that without changing any
    tracker parameter.

    Intentionally a tiny hand parser, not a YAML import: this must
    never be the thing that stops the detector from loading. On any
    problem it returns `default` (0.1, the stock Ultralytics value).
    """
    try:
        with open(path, "r") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                key, sep, rest = stripped.partition(":")
                if not sep or key.strip() != "track_low_thresh":
                    continue
                value = float(rest.split("#", 1)[0].strip())
                if 0.0 <= value <= 1.0:
                    return value
                break
    except (OSError, ValueError):
        pass
    return default


class Detector:
    """
    Wraps a single Ultralytics YOLO model (TensorRT engine or otherwise).

    Three inference paths, all built on the same underlying model:

      - predict(image_bytes)   -> single image, NO tracking.
                                   Used by image detection.
      - predict_frame(frame)   -> single frame, NO tracking.
                                   Kept for backward compatibility --
                                   anything that just wants raw
                                   per-frame detections still works
                                   exactly as before.
      - track_frame(frame)     -> single frame WITH ByteTrack
                                   persistent tracking (adds a stable
                                   `track_id` per physical object).
                                   Used by video + webcam detection.

    track_frame() uses Ultralytics' own tracking support
    (`model.track(..., tracker="bytetrack.yaml")`) rather than a
    homemade tracker, per the requirement to use the standard
    YOLO/Ultralytics + ByteTrack integration. Ultralytics keeps
    tracker state attached to `model.predictor` across calls as long
    as `persist=True` -- call reset_tracker() once at the START of
    every NEW tracking session (new video upload, new "Start Camera"
    click), never per-frame, or track IDs will leak across unrelated
    sessions.

    CONCURRENCY: this Detector instance (and therefore its tracker
    state) is shared across the whole process. Video processing and a
    live webcam session both call track_frame()/reset_tracker() on the
    SAME underlying model + SAME persistent tracker state. Without a
    guard, a video upload and a webcam session running at the same
    moment would corrupt each other's ByteTrack state (interleaved
    track_frame() calls both mutating model.predictor.trackers).
    self._track_lock below serializes every tracker-touching call so
    that a video job and a live webcam session queue up rather than
    race -- this does NOT reload the TensorRT engine (it's still
    loaded exactly once in __init__) and does NOT block image
    detection (predict/predict_frame are stateless and don't touch the
    tracker, so they're intentionally left outside this lock).

    self.tracker_feed_conf: the LOW confidence floor a tracking caller
    should pass into track_frame() as `conf` (see
    _read_tracker_feed_conf). It is NOT the operator/display threshold
    -- the caller applies that itself to the detections track_frame()
    returns.
    """

    def __init__(self, engine_path):
        print("Loading TensorRT model...")
        self.model = YOLO(engine_path)
        self.names = self.model.names
        self._track_lock = threading.Lock()

        # LOW confidence floor for tracking callers to feed into
        # model.track() -- keeps ByteTrack's second-stage association
        # fed. NOT the operator threshold. See _read_tracker_feed_conf.
        self.tracker_feed_conf = _read_tracker_feed_conf(_TRACKER_CONFIG_PATH)

        print(f"Model loaded successfully. Tracker feed conf = {self.tracker_feed_conf:.3f}")

    # ------------------------------------------------------------------
    # Shared, no-tracking inference (image detection, and predict_frame
    # callers that don't need track IDs). Stateless -- no lock needed.
    # ------------------------------------------------------------------
    def _run_inference(self, image):

        if image is None:
            return {
                "success": False,
                "count": 0,
                "detections": [],
                "error": "Invalid image/frame"
            }

        results = self.model.predict(
            source=image,
            verbose=False
        )

        detections = []

        for r in results:
            if r.boxes is None:
                continue

            for b in r.boxes:

                class_id = int(b.cls.item())
                confidence = float(b.conf.item())
                bbox = [int(x) for x in b.xyxy[0].tolist()]

                detections.append({
                    "class": self.names[class_id],
                    "class_id": class_id,
                    "confidence": round(confidence, 4),
                    "bbox": bbox
                })

        return {
            "success": True,
            "count": len(detections),
            "detections": detections
        }

    def predict(self, image_bytes):

        image = cv2.imdecode(
            np.frombuffer(image_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )

        return self._run_inference(image)

    def predict_frame(self, frame):

        return self._run_inference(frame)

    # ------------------------------------------------------------------
    # ByteTrack-based tracking (video + webcam). Every method here
    # touches shared, mutable tracker state and MUST hold
    # self._track_lock for its whole body -- see class docstring.
    # ------------------------------------------------------------------
    def reset_tracker(self):
        """
        Clears any persistent ByteTrack state. Call this exactly once
        at the start of a new tracking session so the new session's
        track IDs start fresh (Drone #1, #2, ...) instead of
        continuing wherever a previous, unrelated session left off.
        """
        with self._track_lock:
            predictor = getattr(self.model, "predictor", None)
            if predictor is not None and hasattr(predictor, "trackers"):
                try:
                    del predictor.trackers
                except Exception:
                    predictor.trackers = None

    def track_frame(self, frame, conf: float = 0.0):
        """
        Same detection shape as predict_frame(), plus a stable
        `track_id` for each physical object, via Ultralytics'
        ByteTrack tracker. `persist=True` keeps the tracker's internal
        state alive across successive calls on this frame stream --
        that persistence is what makes the SAME physical object keep
        the SAME track_id across frames. Call reset_tracker() once per
        new session, never per frame.

        `conf` is passed straight through to model.track() as the
        detection floor Ultralytics applies BEFORE ByteTrack runs.
        Tracking callers (CameraManager) pass the LOW tracker-feed
        floor here (bytetrack.yaml's track_low_thresh, exposed as
        self.tracker_feed_conf), NOT the operator's display threshold,
        so ByteTrack's second-stage association still receives the
        weak boxes it needs to carry an active track through a
        confidence dip instead of losing it and minting a new id on
        recovery. The caller is then responsible for applying the
        real, higher operator threshold to the returned detections --
        that is where "what the operator sees / what gets counted" is
        actually decided. The default 0.0 is for non-webcam callers
        (video) that do their own filtering.

        Holds self._track_lock for the whole call (not just the
        state-mutating part) because model.track() itself reads AND
        writes model.predictor.trackers internally -- a reset_tracker()
        or a second, concurrent track_frame() call from another
        session interleaving partway through would corrupt that state,
        not just race on an assignment.
        """
        if frame is None:
            return {
                "success": False,
                "count": 0,
                "detections": [],
                "error": "Invalid frame"
            }

        with self._track_lock:
            results = self.model.track(
                source=frame,
                persist=True,
                tracker=_TRACKER_CONFIG_PATH,
                conf=conf,
                verbose=False,
            )

        detections = []

        for r in results:
            if r.boxes is None:
                continue

            for b in r.boxes:
                # A box with no assigned ID means ByteTrack hasn't
                # confirmed it as a track yet (e.g. very first frame
                # it appears in). We don't fabricate an ID for it --
                # it simply isn't counted as a unique object until
                # ByteTrack confirms it.
                if b.id is None:
                    continue

                class_id = int(b.cls.item())
                confidence = float(b.conf.item())
                bbox = [int(x) for x in b.xyxy[0].tolist()]
                track_id = int(b.id.item())

                detections.append({
                    "track_id": track_id,
                    "class": self.names[class_id],
                    "class_id": class_id,
                    "confidence": round(confidence, 4),
                    "bbox": bbox
                })

        return {
            "success": True,
            "count": len(detections),
            "detections": detections
        }
