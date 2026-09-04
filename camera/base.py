"""
Camera source abstraction.

Every camera type (USB, CSI, RTSP, HDMI, custom) implements CameraSource.
The rest of the app (inference worker, FastAPI routes) only ever talks
to this interface, never to GStreamer or OpenCV directly, so adding a
new camera type later means adding one new subclass here — nothing
else changes.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .frame_buffer import Frame, LatestFrameBuffer

logger = logging.getLogger("camera")

try:
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst, GLib  # noqa: E402

    Gst.init(None)
    GSTREAMER_AVAILABLE = True
except Exception as exc:  # pragma: no cover - environment dependent
    GSTREAMER_AVAILABLE = False
    _GST_IMPORT_ERROR = exc


class CameraError(RuntimeError):
    """Raised for any camera open/start/read failure. Callers should
    catch this specifically and turn it into a clean API error rather
    than a 500."""


@dataclass
class CameraInfo:
    id: str
    type: str
    name: str
    device: Optional[str] = None
    sensor_id: Optional[int] = None
    url: Optional[str] = None
    resolution: Optional[str] = None
    fps: Optional[float] = None
    extra: dict = field(default_factory=dict)

    def public_dict(self) -> dict:
        """Dict for the frontend — never includes RTSP credentials."""
        d = {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "resolution": self.resolution,
            "fps": self.fps,
        }
        if self.device:
            d["device"] = self.device
        if self.sensor_id is not None:
            d["sensor_id"] = self.sensor_id
        if self.url:
            d["url"] = _redact_credentials(self.url)
        return d


def _redact_credentials(url: str) -> str:
    # rtsp://user:pass@host/... -> rtsp://***:***@host/...
    if "@" not in url:
        return url
    scheme_sep = url.find("://")
    if scheme_sep == -1:
        return url
    scheme = url[: scheme_sep + 3]
    rest = url[scheme_sep + 3 :]
    creds, _, host_and_path = rest.partition("@")
    if ":" in creds:
        return f"{scheme}***:***@{host_and_path}"
    return url


class CameraSource(ABC):
    """
    Base for all camera sources. Concrete sources push frames into
    `self.buffer` (a LatestFrameBuffer) as they arrive; they never get
    pulled synchronously by the inference worker, which decouples
    capture rate from inference rate.
    """

    def __init__(self, info: CameraInfo):
        self.info = info
        self.buffer = LatestFrameBuffer()
        self._running = False
        self._error: Optional[str] = None

    @abstractmethod
    def start(self) -> None:
        """Open the device / start the pipeline. Raise CameraError on failure."""

    @abstractmethod
    def stop(self) -> None:
        """Stop and fully release the device / pipeline. Must be idempotent
        and safe to call even if start() failed partway through."""

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def last_error(self) -> Optional[str]:
        return self._error

    def get_latest_frame(self, timeout: float = 1.0) -> Optional[Frame]:
        return self.buffer.get_latest(block=True, timeout=timeout)


class GStreamerSource(CameraSource):
    """
    Shared plumbing for every GStreamer-backed source: builds the
    pipeline from a string, wires an appsink callback that decodes each
    sample straight into a numpy BGR frame and pushes it into the
    LatestFrameBuffer, and watches the bus for errors/EOS on a
    dedicated GLib mainloop thread (needed because appsink's own
    'new-sample' signal fires on GStreamer's streaming thread, but bus
    messages need a mainloop pumping them).
    """

    def __init__(self, info: CameraInfo, pipeline_str):
        super().__init__(info)
        if not GSTREAMER_AVAILABLE:
            raise CameraError(
                "GStreamer Python bindings (gi / Gst) are not available: "
                f"{_GST_IMPORT_ERROR}. Install python3-gi and gir1.2-gstreamer-1.0."
            )
        # pipeline_str may be a single pipeline string, or a list of
        # candidate pipeline strings to try in order (first one that
        # reaches PLAYING wins). Most sources still pass a single str;
        # V4L2CameraSource passes a fallback chain since USB webcam
        # capabilities vary wildly device to device.
        self._pipeline_candidates = [pipeline_str] if isinstance(pipeline_str, str) else list(pipeline_str)
        self.pipeline_str = self._pipeline_candidates[0]
        self._pipeline: Optional["Gst.Pipeline"] = None
        self._appsink = None
        self._loop: Optional["GLib.MainLoop"] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._width: Optional[int] = None
        self._height: Optional[int] = None

    def start(self) -> None:
        if self._running:
            return

        errors = []
        for idx, candidate in enumerate(self._pipeline_candidates):
            try:
                self._start_pipeline(candidate)
                if idx > 0:
                    logger.warning(
                        "Camera %s: candidate pipeline #%d succeeded after %d earlier "
                        "attempt(s) failed (device likely doesn't support the requested "
                        "resolution/format/fps exactly).",
                        self.info.id, idx + 1, idx,
                    )
                return
            except CameraError as exc:
                errors.append(str(exc))
                self._teardown_pipeline()
                continue

        raise CameraError(
            f"GStreamer pipeline for {self.info.id} failed to reach PLAYING state after "
            f"trying {len(self._pipeline_candidates)} pipeline variant(s). Check that the "
            "required plugins/elements are installed (run: gst-inspect-1.0 <element>), the "
            "device/URL is correct, and the device isn't already in use by another process "
            "(lsof /dev/videoX). Attempts:\n" + "\n".join(f"  [{i+1}] {e}" for i, e in enumerate(errors))
        )

    def _start_pipeline(self, pipeline_str: str) -> None:
        logger.info("Starting GStreamer pipeline for %s: %s", self.info.id, pipeline_str)
        try:
            self._pipeline = Gst.parse_launch(pipeline_str)
        except GLib.Error as exc:
            raise CameraError(f"Failed to parse GStreamer pipeline: {exc}") from exc

        self._appsink = self._pipeline.get_by_name("sink")
        if self._appsink is None:
            raise CameraError(
                "Pipeline must contain an appsink element named 'sink' "
                "(add '... ! appsink name=sink')."
            )
        self._appsink.set_property("emit-signals", True)
        self._appsink.set_property("max-buffers", 1)
        self._appsink.set_property("drop", True)
        self._appsink.set_property("sync", False)
        self._appsink.connect("new-sample", self._on_new_sample)

        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        self._loop = GLib.MainLoop()
        self._loop_thread = threading.Thread(target=self._loop.run, daemon=True)
        self._loop_thread.start()

        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise CameraError(
                f"'{pipeline_str}' failed to reach PLAYING state synchronously "
                "(usually a caps negotiation failure — the device doesn't support the "
                "requested width/height/framerate/format combination, or is busy)."
            )

        # Wait briefly for the pipeline to actually reach PLAYING so start()
        # fails fast instead of the caller finding out 5s later via /status.
        state_change_ok, _, _ = self._pipeline.get_state(timeout=5 * Gst.SECOND)
        if state_change_ok == Gst.StateChangeReturn.FAILURE:
            raise CameraError(f"'{pipeline_str}' failed during PREROLL.")
        if state_change_ok != Gst.StateChangeReturn.SUCCESS:
            # ASYNC/NO_PREROLL after the timeout means it never actually got
            # there either — treat as failure so we fall through to the next
            # candidate instead of reporting a fake success.
            raise CameraError(
                f"'{pipeline_str}' did not reach PLAYING within 5s "
                f"(state_change_return={state_change_ok})."
            )

        self._running = True
        self._error = None

    def stop(self) -> None:
        self._running = False
        self._teardown_pipeline()
        self.buffer.clear()

    def _teardown_pipeline(self) -> None:
        """Tear down whatever partial pipeline/loop state a failed or
        successful start() left behind. Safe to call multiple times and
        safe to call after a start() attempt failed partway through —
        used both by stop() and by start()'s fallback-candidate loop."""
        if self._pipeline is not None:
            try:
                self._pipeline.set_state(Gst.State.NULL)
            except Exception:
                logger.exception("Error stopping GStreamer pipeline for %s", self.info.id)
            self._pipeline = None
        self._appsink = None
        if self._loop is not None:
            try:
                self._loop.quit()
            except Exception:
                pass
            self._loop = None
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=2)
            self._loop_thread = None

    def _on_bus_message(self, bus, message):  # noqa: ANN001
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            self._error = f"{err.message} ({debug})" if debug else err.message
            logger.error("GStreamer error on %s: %s", self.info.id, self._error)
            self._running = False
        elif t == Gst.MessageType.EOS:
            self._error = "Stream ended unexpectedly (EOS)."
            logger.warning("GStreamer EOS on %s", self.info.id)
            self._running = False
        elif t == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            logger.warning("GStreamer warning on %s: %s (%s)", self.info.id, warn.message, debug)

    def _on_new_sample(self, sink):  # noqa: ANN001
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR
        buf = sample.get_buffer()
        caps = sample.get_caps()
        struct = caps.get_structure(0)
        width = struct.get_value("width")
        height = struct.get_value("height")
        self._width, self._height = width, height

        ok, map_info = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.ERROR
        try:
            # Pipelines always terminate in "video/x-raw,format=BGR" so this
            # reshape is safe and needs no color conversion in Python.
            frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape((height, width, 3)).copy()
        finally:
            buf.unmap(map_info)

        self.buffer.put(frame)
        return Gst.FlowReturn.OK

    @property
    def resolution(self) -> Optional[str]:
        if self._width and self._height:
            return f"{self._width}x{self._height}"
        return None
