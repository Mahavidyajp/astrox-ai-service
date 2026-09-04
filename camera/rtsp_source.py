"""
RTSP/IP camera source.

Pipeline (H.264 stream, the common case):
    rtspsrc location=URL latency=L protocols=tcp
        ! rtph264depay
        ! h264parse
        ! nvv4l2decoder
        ! nvvidconv
        ! video/x-raw,format=BGRx
        ! videoconvert
        ! video/x-raw,format=BGR
        ! appsink name=sink

Latency choices (explained per-property since this is the highest-risk
source for latency, per your requirements):
- latency=L (ms): this is rtspsrc's internal jitterbuffer size. It is
  NOT free to set to 0 — RTP over UDP can arrive out of order, and a
  jitterbuffer is what re-orders/smooths that. Too low -> visible
  corruption/frame drops on a lossy Wi-Fi link; too high -> added
  latency. Default here is 100ms, exposed via config as
  `rtsp_latency_ms` so it can be tuned per-camera/per-network. This is
  the single biggest latency knob for RTSP.
- protocols=tcp: forces RTP-over-TCP instead of the default UDP+RTCP.
  Counter-intuitively this is often *lower effective latency* on
  networks with any packet loss, because lost UDP packets otherwise
  stall the jitterbuffer waiting for retransmission that never comes,
  whereas TCP just resends the missing segment inline. If your camera
  is on a clean wired LAN, protocols=udp with a lower `latency` value
  can beat this — exposed via config as `rtsp_protocol`.
- nvv4l2decoder: Jetson's hardware H.264/H.265 decoder. Software
  decode (avdec_h264) would be a CPU bottleneck competing with YOLO
  inference for the same cores.
- nvvidconv / videoconvert: same NVMM -> BGR conversion as the CSI
  path, for the same reason.
- No extra `queue` elements: same drop=true/max-buffers=1 appsink
  policy applies — a stalled/slow network write should never let
  frames pile up before appsink.

H.265 support: pass codec="h265" in config to use rtph265depay/h265parse
instead. MJPEG-over-RTSP cameras are not covered here (rare); add a
branch for rtpjpegdepay/jpegdec if needed.
"""

from __future__ import annotations

from .base import CameraInfo, GStreamerSource

_DEPAY = {
    "h264": "rtph264depay ! h264parse",
    "h265": "rtph265depay ! h265parse",
}


def _build_pipeline(
    url: str,
    latency_ms: int,
    protocol: str,
    codec: str,
) -> str:
    if codec not in _DEPAY:
        raise ValueError(f"Unsupported RTSP codec '{codec}', expected one of {list(_DEPAY)}")
    protocols_prop = f"protocols={protocol}" if protocol else ""
    return (
        f'rtspsrc location="{url}" latency={latency_ms} {protocols_prop} ! '
        f"{_DEPAY[codec]} ! "
        f"nvv4l2decoder ! "
        f"nvvidconv ! video/x-raw,format=BGRx ! "
        f"videoconvert ! video/x-raw,format=BGR ! "
        f"appsink name=sink"
    )


class RTSPCameraSource(GStreamerSource):
    def __init__(
        self,
        info: CameraInfo,
        url: str,
        latency_ms: int = 100,
        protocol: str = "tcp",
        codec: str = "h264",
    ):
        pipeline_str = _build_pipeline(url, latency_ms, protocol, codec)
        super().__init__(info, pipeline_str)
