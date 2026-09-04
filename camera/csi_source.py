"""
Jetson CSI camera source (nvarguscamerasrc).

Pipeline:
    nvarguscamerasrc sensor-id=N
        ! video/x-raw(memory:NVMM),width=W,height=H,framerate=F/1,format=NV12
        ! nvvidconv
        ! video/x-raw,format=BGRx
        ! videoconvert
        ! video/x-raw,format=BGR
        ! appsink name=sink

Notes:
- Capture happens in NVMM (NVIDIA's hardware memory) so the ISP writes
  directly into GPU-backed memory instead of system RAM — this is the
  hardware-accelerated part of the pipeline.
- nvvidconv is also hardware-accelerated on Jetson and does the
  NVMM -> system-memory copy plus the NV12 -> BGRx conversion in one
  step; it's the standard/only supported way to get argus camera data
  out of NVMM.
- A second, CPU-side videoconvert (BGRx -> BGR) is required because
  nvvidconv cannot itself produce packed BGR (only BGRx/RGBA family in
  system memory); this is one extra cheap conversion on a 3-channel
  vs 4-channel buffer, not a bottleneck at 720p/1080p.
- No `queue` inserted for the same reason as v4l2 sources: appsink's
  drop=true/max-buffers=1 is the only buffering point we want.
"""

from __future__ import annotations

from .base import CameraInfo, GStreamerSource


def _build_pipeline(sensor_id: int, width: int, height: int, fps: int) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM),width={width},height={height},framerate={fps}/1,format=NV12 ! "
        f"nvvidconv ! video/x-raw,format=BGRx ! "
        f"videoconvert ! video/x-raw,format=BGR ! "
        f"appsink name=sink"
    )


class CSICameraSource(GStreamerSource):
    def __init__(
        self,
        info: CameraInfo,
        sensor_id: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ):
        pipeline_str = _build_pipeline(sensor_id, width, height, fps)
        super().__init__(info, pipeline_str)
