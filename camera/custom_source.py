"""
Escape hatch for camera types not covered by USB/CSI/RTSP/HDMI.

The operator supplies a raw GStreamer pipeline string in cameras.yaml
under `pipeline:`. It MUST end in an appsink named 'sink' (see base.py
GStreamerSource for why) and should produce video/x-raw,format=BGR.

Example config entry:
    - id: custom-thermal-0
      type: CUSTOM
      name: "Thermal camera"
      pipeline: >
        v4l2src device=/dev/video4 !
        video/x-raw,format=GRAY16_LE,width=640,height=512 !
        videoconvert ! video/x-raw,format=BGR !
        appsink name=sink
"""

from __future__ import annotations

from .base import CameraInfo, GStreamerSource


class CustomCameraSource(GStreamerSource):
    def __init__(self, info: CameraInfo, pipeline: str):
        if "appsink" not in pipeline or "name=sink" not in pipeline:
            raise ValueError(
                "Custom pipeline must contain '... ! appsink name=sink' so the "
                "capture layer can pull frames from it."
            )
        super().__init__(info, pipeline)
