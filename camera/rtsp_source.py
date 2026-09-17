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

from typing import Optional
from urllib.parse import quote

from .base import CameraInfo, GStreamerSource

_DEPAY = {
    "h264": "rtph264depay ! h264parse",
    "h265": "rtph265depay ! h265parse",
}


def build_rtsp_url(url: str, username: Optional[str] = None, password: Optional[str] = None) -> str:
    """Combine a bare RTSP URL with optional, separately-supplied
    username/password into rtsp://user:pass@host/... .

    This exists so the "Connect Drone" frontend form (and the
    /webcam/rtsp/test and /webcam/rtsp/connect routes) never has to
    build a credentialed URL itself — username and password stay two
    separate fields end-to-end until this function combines them,
    once, server-side.

    - No `username` -> `url` is returned exactly as given, so a stream
      with no auth (or one where the operator already pasted a
      pre-composed rtsp://user:pass@host URL) works unchanged.
    - `username` given -> any credentials already embedded in `url`
      are replaced (not doubled up) with the supplied ones.
    - Both parts are percent-encoded (urllib.parse.quote) so a
      password containing "@", ":" or "/" can't corrupt the URL or be
      misread as part of the host/path.

    Raises ValueError if `url` has no "scheme://" — fails fast rather
    than silently building a malformed pipeline location.
    """
    if not username:
        return url

    scheme_sep = url.find("://")
    if scheme_sep == -1:
        raise ValueError(f"Invalid RTSP URL (missing scheme): {url!r}")
    scheme = url[: scheme_sep + 3]
    rest = url[scheme_sep + 3 :]
    if "@" in rest:
        rest = rest.split("@", 1)[1]

    user_enc = quote(username, safe="")
    if password:
        return f"{scheme}{user_enc}:{quote(password, safe='')}@{rest}"
    return f"{scheme}{user_enc}@{rest}"


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
