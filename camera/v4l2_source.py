"""
USB webcam and HDMI capture card sources.

Both are plain V4L2 devices from GStreamer's point of view, so they
share one implementation; only the CameraInfo.type differs (kept
separate for discovery/labeling purposes).

Fallback chain (see GStreamerSource.start() in base.py):
    1. MJPEG at the requested width/height/fps, decoded on the CPU.
    2. Raw YUYV/etc at the requested width/height/fps.
    3. Native/adaptive: no caps forced on v4l2src at all — it
       negotiates whatever format+resolution+fps the device offers by
       default, decodebin figures out whether that needs a JPEG
       decode or is already raw, and videoscale resizes to the
       requested output size afterwards.

Candidates 1 and 2 are attempted first because they're cheaper (the
device does more of the work, and no CPU rescale is needed) and give
predictable output resolution. But not every camera supports an
arbitrary (width, height, fps) triple you might request — cheap UVC
sensors commonly only do e.g. 1280x720 at 5-10fps in MJPEG, or only
support a couple of fixed resolutions. Forcing an unsupported triple
makes v4l2src fail caps negotiation immediately, which surfaces as
"failed to reach PLAYING state" even though the device is perfectly
fine — candidate 3 is what makes such devices work at all, at
whatever fps/resolution they actually support.

Latency choices:
- io-mode=2 (mmap): avoids an extra memcpy vs. read() mode.
- No `queue` element is inserted between v4l2src and appsink: adding
  a queue here would only add buffering/latency for no benefit, since
  appsink itself is configured for drop=true, max-buffers=1.
- videoconvert output is pinned to BGR so detector.predict_frame()
  and cv2 drawing get exactly the layout they already expect —
  zero-conversion in Python. _on_new_sample() reads width/height off
  each sample's own caps, so a variable/adaptive output size (as
  candidate 3 can produce if videoscale is skipped) is handled fine.
"""

from __future__ import annotations

from typing import List

from .base import CameraInfo, GStreamerSource


def _build_pipeline_candidates(
    device: str, width: int, height: int, fps: int, use_mjpeg: bool
) -> List[str]:
    candidates = []

    if use_mjpeg:
        candidates.append(
            f"v4l2src device={device} io-mode=2 ! "
            f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
            f"jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
            f"appsink name=sink"
        )

    candidates.append(
        f"v4l2src device={device} io-mode=2 ! "
        f"video/x-raw,width={width},height={height},framerate={fps}/1 ! "
        f"videoconvert ! video/x-raw,format=BGR ! "
        f"appsink name=sink"
    )

    # Universal fallback: don't force any caps on v4l2src at all, so it
    # just uses the device's default format/resolution/fps (whatever
    # that negotiates to). decodebin transparently handles either raw
    # or MJPEG output from v4l2src. videoscale then resizes to the
    # requested output size in software, which always succeeds.
    candidates.append(
        f"v4l2src device={device} io-mode=2 ! "
        f"decodebin ! videoconvert ! videoscale ! "
        f"video/x-raw,format=BGR,width={width},height={height} ! "
        f"appsink name=sink"
    )

    return candidates


class V4L2CameraSource(GStreamerSource):
    """Covers both USB webcams and HDMI capture devices (type is just a label)."""

    def __init__(
        self,
        info: CameraInfo,
        device: str,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        use_mjpeg: bool = True,
    ):
        pipeline_candidates = _build_pipeline_candidates(device, width, height, fps, use_mjpeg)
        super().__init__(info, pipeline_candidates)
