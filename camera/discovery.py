"""
Camera discovery.

USB/HDMI: enumerated live via v4l2-ctl (falls back to a plain
/dev/video* scan with OpenCV probing if v4l2-ctl isn't installed).

CSI: NOT auto-detected. Argus can't be probed non-destructively while
another process might be using it, and Jetson CSI camera counts/types
are a hardware fact of the carrier board — so CSI entries come from
config (cameras.yaml), same as RTSP.

RTSP: never auto-discovered (there is no safe/generic way to discover
arbitrary IP cameras on a network); always explicit config entries.
"""

from __future__ import annotations

import logging
import re
import subprocess
from typing import List

from .base import CameraInfo
from .config import CameraConfig, load_camera_config

logger = logging.getLogger("camera.discovery")


def _probe_v4l2_devices() -> List[dict]:
    """Returns [{device, name}] for every /dev/videoN with video-capture
    capability, using v4l2-ctl --list-devices when available."""
    try:
        output = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.warning("v4l2-ctl not found or timed out; falling back to /dev/video* scan.")
        return _probe_v4l2_devices_fallback()

    devices = []
    current_name = None
    for line in output.splitlines():
        if not line.strip():
            current_name = None
            continue
        if not line.startswith("\t") and not line.startswith(" "):
            # e.g. "UVC Camera (046d:0825) (usb-3610000.usb-2.3):"
            current_name = re.sub(r"\s*\(.*?\):?\s*$", "", line).strip()
            continue
        dev_path = line.strip()
        if dev_path.startswith("/dev/video"):
            devices.append({"device": dev_path, "name": current_name or dev_path})
    return devices


def _probe_v4l2_devices_fallback() -> List[dict]:
    import os

    devices = []
    for i in range(10):
        path = f"/dev/video{i}"
        if os.path.exists(path):
            devices.append({"device": path, "name": f"Video device {i}"})
    return devices


def _is_capture_capable(device: str) -> bool:
    """
    Many UVC webcams expose a *second* /dev/videoN node alongside the
    real capture node — e.g. a metadata-only node for embedded
    timestamps/IMU data on some depth/IR cameras. v4l2-ctl
    --list-devices lists both under the same physical device with no
    way to tell them apart from the name alone, so a naive "usb-{idx}"
    enumeration can hand out an id that points at a node that will
    NEVER reach PLAYING no matter what pipeline you throw at it.

    This checks the node's reported device capabilities and keeps only
    ones that actually advertise video capture (not just metadata
    capture or output-only).
    """
    try:
        output = subprocess.run(
            ["v4l2-ctl", "-d", device, "-D"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # Can't check — assume capture-capable rather than silently
        # dropping cameras when v4l2-ctl -D isn't available.
        return True

    # Look at the "Device Caps" block specifically (not "Driver Caps",
    # which lists everything the driver supports across all its nodes,
    # not just this one).
    in_device_caps = False
    for line in output.splitlines():
        if "Device Caps" in line:
            in_device_caps = True
            continue
        if in_device_caps:
            if not line.startswith("\t") and not line.startswith(" "):
                break
            if "Video Capture" in line:
                return True
    return False


def discover_usb_and_hdmi() -> List[CameraInfo]:
    """
    Filters to nodes that actually advertise V4L2 "Video Capture"
    capability (see _is_capture_capable) so a camera's metadata/control
    node never silently steals a usb-N id from the real capture node —
    that mismatch is what makes a given id fail to reach PLAYING no
    matter what pipeline is used against it.
    """
    raw = _probe_v4l2_devices()
    capture_only = [dev for dev in raw if _is_capture_capable(dev["device"])]
    infos = []
    for idx, dev in enumerate(capture_only):
        infos.append(
            CameraInfo(
                id=f"usb-{idx}",
                type="USB",
                name=dev["name"],
                device=dev["device"],
            )
        )
    return infos


def discover_all(config: CameraConfig | None = None) -> List[CameraInfo]:
    if config is None:
        config = load_camera_config()

    cameras: List[CameraInfo] = []

    if config.auto_discover_usb:
        cameras.extend(discover_usb_and_hdmi())

    for entry in config.csi_cameras:
        cameras.append(
            CameraInfo(
                id=entry["id"],
                type="CSI",
                name=entry.get("name", f"CSI Camera {entry.get('sensor_id', 0)}"),
                sensor_id=entry.get("sensor_id", 0),
                resolution=f"{entry.get('width', 1280)}x{entry.get('height', 720)}",
                fps=entry.get("fps", 30),
            )
        )

    for entry in config.rtsp_cameras:
        cameras.append(
            CameraInfo(
                id=entry["id"],
                type="RTSP",
                name=entry.get("name", "IP Camera"),
                url=entry["url"],
            )
        )

    for entry in config.custom_cameras:
        cameras.append(
            CameraInfo(
                id=entry["id"],
                type="CUSTOM",
                name=entry.get("name", entry["id"]),
            )
        )

    return cameras
