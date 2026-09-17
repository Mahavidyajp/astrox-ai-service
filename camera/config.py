from __future__ import annotations

import logging
import os
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger("camera.config")

DEFAULT_CONFIG_PATH = os.getenv("CAMERAS_CONFIG_PATH", "cameras.yaml")

# Guards the read-modify-write in upsert_rtsp_camera() so two concurrent
# "Connect Drone" requests can't race and silently drop one write.
_write_lock = threading.Lock()


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


def _write_camera_config_dict(raw: Dict[str, Any], path: str) -> None:
    """Atomically write `raw` back to `path` as YAML.

    cameras.yaml is reloaded fresh on every /webcam/cameras and
    /webcam/start call (see load_camera_config's callers in
    manager.py), so a partially-written file would be picked up
    immediately by the next request. Writing to a temp file in the
    same directory and os.replace()-ing it into place makes the
    on-disk file always either the old complete version or the new
    complete version, never a truncated one.

    Note: PyYAML's safe_dump does not preserve comments or formatting,
    so any comments in a hand-edited cameras.yaml are lost the first
    time this function writes to it.
    """
    import yaml

    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".cameras-", suffix=".yaml", dir=directory)
    try:
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(raw, f, default_flow_style=False, sort_keys=False)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def upsert_rtsp_camera(entry: Dict[str, Any], path: str = DEFAULT_CONFIG_PATH) -> None:
    """Add or replace one entry in cameras.yaml's `rtsp_cameras` list,
    matched by `entry["id"]`. Every other camera and every other
    top-level key (auto_discover_usb, csi_cameras, custom_cameras,
    usb_cameras) is left exactly as it was.

    `entry["url"]` may already have credentials embedded
    (rtsp://user:pass@host/...) — this function just persists whatever
    it's given; composing that URL from separate username/password
    fields is the caller's job (the /webcam/rtsp/connect route), kept
    out of this module so the write path here never has to reason
    about credentials. It never logs `entry` itself, only the id/name,
    for the same reason.

    Raises ValueError if `entry` is missing `id` or `url` — fails
    fast rather than writing a camera the rest of the system can't
    resolve or start.
    """
    import yaml

    camera_id = entry.get("id")
    if not camera_id:
        raise ValueError("upsert_rtsp_camera: entry must have an 'id'.")
    if not entry.get("url"):
        raise ValueError("upsert_rtsp_camera: entry must have a 'url'.")

    with _write_lock:
        if os.path.exists(path):
            with open(path, "r") as f:
                raw = yaml.safe_load(f) or {}
        else:
            raw = {"auto_discover_usb": True}

        rtsp_cameras = list(raw.get("rtsp_cameras") or [])
        replaced = False
        for i, existing in enumerate(rtsp_cameras):
            if existing.get("id") == camera_id:
                rtsp_cameras[i] = entry
                replaced = True
                break
        if not replaced:
            rtsp_cameras.append(entry)

        raw["rtsp_cameras"] = rtsp_cameras
        _write_camera_config_dict(raw, path)

    logger.info(
        "cameras.yaml: %s RTSP camera '%s' (%s)",
        "updated" if replaced else "added",
        camera_id,
        entry.get("name", camera_id),
    )
