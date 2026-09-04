from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger("camera.config")

DEFAULT_CONFIG_PATH = os.getenv("CAMERAS_CONFIG_PATH", "cameras.yaml")


@dataclass
class CameraConfig:
    auto_discover_usb: bool = True
    csi_cameras: List[Dict[str, Any]] = field(default_factory=list)
    rtsp_cameras: List[Dict[str, Any]] = field(default_factory=list)
    custom_cameras: List[Dict[str, Any]] = field(default_factory=list)
    # Per-device overrides for auto-discovered USB/HDMI cameras. Matched
    # by `device` path (e.g. "/dev/video0"), NOT by usb-N id, since the
    # id is assigned dynamically at discovery time and can shift if
    # cameras are plugged in a different order. Keys: device (required),
    # width, height, fps, use_mjpeg. Anything not overridden falls back
    # to CameraManager's defaults (1280x720@30 MJPEG).
    usb_cameras: List[Dict[str, Any]] = field(default_factory=list)


def load_camera_config(path: str = DEFAULT_CONFIG_PATH) -> CameraConfig:
    if not os.path.exists(path):
        logger.warning("No cameras.yaml found at %s — USB/HDMI auto-discovery only.", path)
        return CameraConfig()

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "cameras.yaml exists but PyYAML is not installed. Run: "
            "pip install pyyaml --break-system-packages"
        ) from exc

    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    return CameraConfig(
        auto_discover_usb=raw.get("auto_discover_usb", True),
        csi_cameras=raw.get("csi_cameras", []) or [],
        rtsp_cameras=raw.get("rtsp_cameras", []) or [],
        custom_cameras=raw.get("custom_cameras", []) or [],
        usb_cameras=raw.get("usb_cameras", []) or [],
    )
