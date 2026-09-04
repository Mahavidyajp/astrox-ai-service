"""
MediaMTX publisher.

Streams the live RAW camera frames to a local MediaMTX server as an
H.264 / RTSP stream (appsrc -> videoconvert/scale -> openh264enc ->
h264parse -> rtspclientsink). MediaMTX hands that to browsers over
WebRTC (WHEP).

Runs entirely on ONE background thread (_feed_loop): it (re)builds the
pipeline, pushes frames into appsrc on a fixed cadence, and polls the
GStreamer bus for errors/EOS -- all in the same loop. Deliberately NO
GLib.MainLoop: the camera sources already run their own GLib main loop,
and a second one on the default main context from another thread
deadlocks the GIL and freezes the whole process. Bus polling avoids
that. start() only spawns the thread and returns, so it can never block
a request handler even if MediaMTX is unreachable.

On any pipeline error (MediaMTX down, RTSP SETUP racing the first
encoded frame) the loop tears the pipeline down and rebuilds it after a
~1s backoff. It also primes the encoder with a few frames before
returning from a build, so rtspclientsink's ANNOUNCE/SETUP sees valid
H.264 caps on the first try. None of this can stall or kill the
inference worker -- they share nothing but the frame buffer, which this
only reads.

The Jetson Orin Nano has no hardware video encoder, so encode is
openh264enc on the CPU (~1 core at 960x540 / 25 fps).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger("camera.publisher")

try:
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst  # noqa: E402

    Gst.init(None)  # safe to call more than once
    _GST_OK = True
except Exception as exc:  # pragma: no cover - environment dependent
    _GST_OK = False
    _GST_ERR = exc


# Backoff before rebuilding the pipeline after an error / a failed build.
_REBUILD_AFTER_ERROR_S = 1.0
_REBUILD_AFTER_BUILD_FAIL_S = 1.5


class MediaMTXPublisher:
    def __init__(
        self,
        rtsp_url: str,
        width: int = 960,
        height: int = 540,
        fps: int = 25,
        bitrate_bps: int = 2_500_000,
    ) -> None:
        self._rtsp_url = rtsp_url
        self._w = int(width)
        self._h = int(height)
        self._fps = max(1, int(fps))
        self._bitrate = int(bitrate_bps)

        self._get_frame: Optional[Callable[[], Optional["np.ndarray"]]] = None
        self._pipeline = None
        self._appsrc = None
        self._bus = None
        self._feed_thread: Optional[threading.Thread] = None

        self._running = False
        self._lock = threading.Lock()
        self._src_caps_set = False
        self._last_frame: Optional["np.ndarray"] = None
        self._next_rebuild_at = 0.0

    @property
    def enabled(self) -> bool:
        return _GST_OK

    # ------------------------------------------------------------------
    # Public lifecycle -- start() never blocks the caller.
    # ------------------------------------------------------------------
    def start(self, get_frame: Callable[[], Optional["np.ndarray"]]) -> None:
        if not _GST_OK:
            logger.warning(
                "GStreamer not available (%s); MediaMTX publisher disabled, "
                "detection is unaffected.", _GST_ERR,
            )
            return
        with self._lock:
            if self._running:
                return
            self._running = True
            self._get_frame = get_frame
            self._last_frame = None
            self._next_rebuild_at = 0.0

        self._feed_thread = threading.Thread(
            target=self._feed_loop, name="mediamtx-feed", daemon=True
        )
        self._feed_thread.start()
        logger.info(
            "MediaMTX publisher started -> %s  (%dx%d @ %d fps, %d bps)",
            self._rtsp_url, self._w, self._h, self._fps, self._bitrate,
        )

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False
        if self._feed_thread is not None:
            self._feed_thread.join(timeout=4)
            self._feed_thread = None
        self._teardown_pipeline()
        logger.info("MediaMTX publisher stopped.")

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    def _pipeline_str(self) -> str:
        return (
            "appsrc name=src is-live=true do-timestamp=true format=time block=false "
            "! queue max-size-buffers=4 leaky=downstream "
            "! videoconvert ! videoscale "
            f"! video/x-raw,format=I420,width={self._w},height={self._h},framerate={self._fps}/1 "
            "! openh264enc rate-control=bitrate "
            f"bitrate={self._bitrate} gop-size={self._fps * 2} complexity=low "
            "! h264parse config-interval=1 "
            f"! rtspclientsink name=sink location={self._rtsp_url} protocols=tcp"
        )

    def _build_pipeline(self) -> bool:
        pstr = self._pipeline_str()
        try:
            self._pipeline = Gst.parse_launch(pstr)
        except Exception:
            logger.exception("MediaMTX publisher: parse_launch failed for: %s", pstr)
            self._pipeline = None
            return False

        self._appsrc = self._pipeline.get_by_name("src")
        self._bus = self._pipeline.get_bus()
        self._src_caps_set = False

        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            logger.error("MediaMTX publisher: pipeline failed to start.")
            self._teardown_pipeline()
            return False

        # Prime the encoder with a few frames BEFORE returning, so the
        # H.264 caps (SPS/PPS) exist by the time rtspclientsink does its
        # RTSP ANNOUNCE/SETUP. Without this the first build races the
        # first encoded frame and fails with
        # "Could not get/set settings ... setup_streams".
        if self._last_frame is not None and self._appsrc is not None:
            self._ensure_src_caps(self._last_frame)
            try:
                data = np.ascontiguousarray(self._last_frame).tobytes()
                for _ in range(8):
                    self._appsrc.emit("push-buffer", Gst.Buffer.new_wrapped(data))
                    time.sleep(0.03)
            except Exception:
                logger.exception("MediaMTX publisher: priming push-buffer failed")

        logger.info("MediaMTX publisher: pipeline PLAYING -> %s", self._rtsp_url)
        return True

    def _teardown_pipeline(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.set_state(Gst.State.NULL)
            except Exception:
                logger.exception("MediaMTX publisher: error tearing down pipeline")
            self._pipeline = None
        self._appsrc = None
        self._bus = None
        self._src_caps_set = False

    def _poll_bus(self) -> None:
        """Non-blocking bus drain. On ERROR/EOS, drop the pipeline and
        schedule a rebuild after a short backoff."""
        if self._bus is None:
            return
        while True:
            msg = self._bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
            if msg is None:
                return
            if msg.type == Gst.MessageType.ERROR:
                err, dbg = msg.parse_error()
                logger.error("MediaMTX publisher GStreamer error: %s (%s)", err.message, dbg)
            else:
                logger.warning("MediaMTX publisher: EOS.")
            self._teardown_pipeline()
            self._next_rebuild_at = time.monotonic() + _REBUILD_AFTER_ERROR_S
            return

    # ------------------------------------------------------------------
    # Frame feeding
    # ------------------------------------------------------------------
    def _ensure_src_caps(self, frame: "np.ndarray") -> None:
        if self._src_caps_set or self._appsrc is None:
            return
        h, w = frame.shape[:2]
        caps = Gst.Caps.from_string(
            f"video/x-raw,format=BGR,width={w},height={h},framerate={self._fps}/1"
        )
        self._appsrc.set_property("caps", caps)
        self._src_caps_set = True

    def _feed_loop(self) -> None:
        period = 1.0 / float(self._fps)
        next_t = time.monotonic()
        while True:
            with self._lock:
                if not self._running:
                    break

            # 1. Grab the latest camera frame (blocks up to ~1s inside
            #    get_frame). Keep the last one to hold cadence / prime.
            frame = None
            try:
                if self._get_frame is not None:
                    frame = self._get_frame()
            except Exception:
                logger.exception("MediaMTX publisher: get_frame() failed")
                frame = None
            if frame is not None:
                self._last_frame = frame

            # 2. (Re)build the pipeline once we actually have a frame to
            #    prime it with and the backoff has elapsed.
            now = time.monotonic()
            if (
                self._pipeline is None
                and self._last_frame is not None
                and now >= self._next_rebuild_at
            ):
                if not self._build_pipeline():
                    self._next_rebuild_at = time.monotonic() + _REBUILD_AFTER_BUILD_FAIL_S

            # 3. Handle any pipeline error/EOS.
            self._poll_bus()

            # 4. Push the current frame.
            appsrc = self._appsrc
            if self._last_frame is not None and appsrc is not None:
                try:
                    self._ensure_src_caps(self._last_frame)
                    data = np.ascontiguousarray(self._last_frame).tobytes()
                    appsrc.emit("push-buffer", Gst.Buffer.new_wrapped(data))
                except Exception:
                    logger.exception("MediaMTX publisher: push-buffer failed")

            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()
