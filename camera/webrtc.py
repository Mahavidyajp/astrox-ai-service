"""
WebRTC video transport.

Delivers RAW (undecorated) camera frames to the browser as a low-latency
H.264/WebRTC video track. This is intentionally decoupled from
detections: the video track never draws boxes into the frame, and
knows nothing about the detector. Detection results travel on their own
channel (the `/webcam/ws/detections` WebSocket, wired up in app.py) and
are rendered by the frontend as a <canvas> overlay on top of the
<video> element. This split is what lets each transport be tuned/scaled
independently -- e.g. a slow inference model no longer has any way to
add latency to the video path, and vice versa.

Why WebRTC instead of MJPEG here:
- H.264 (hardware-encoded where the platform supports it, software x264
  otherwise, both handled by aiortc/PyAV) sends far fewer bytes per
  frame than full baseline JPEGs pushed over a chunked HTTP response.
- The browser's own RTP jitter buffer and A/V sync machinery is
  purpose-built for live video; MJPEG-over-HTTP has none of that.
- A <video> element with a WebRTC track gets real hardware-accelerated
  decode in the browser, instead of the browser re-decoding a JPEG into
  a bitmap on every single frame via <img>.

Latest-frame semantics (never let a queue build up):
- LatestFrameVideoTrack never blocks waiting for a new frame. Every
  time the RTP sender pulls the next frame -- paced by
  VideoStreamTrack.next_timestamp(), which self-paces to ~30fps against
  the wall clock -- it does a *non-blocking* read of whatever the
  newest frame in CameraManager's buffer is right now and encodes that.
  If nothing newer has arrived since the last pull, the previous frame
  is simply repeated (a normal, standard thing for any webcam feed to
  do) instead of blocking the encoder or letting frames pile up.
- Every WebRTC viewer gets its own `consumer_id` into the shared
  LatestFrameBuffer (see frame_buffer.py's multi-consumer support), so
  concurrent viewers -- and the inference worker -- each independently
  see "the latest frame relative to what I've already read", without
  stealing frames from one another.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import List, Optional, Set

import numpy as np
from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    VideoStreamTrack,
)
from av import VideoFrame

logger = logging.getLogger("camera.webrtc")

# Shown only until the first real captured frame arrives (e.g. while
# the pipeline is still negotiating caps), so the video track always has
# *something* valid to hand the encoder immediately on connect.
_BLANK_FRAME_SIZE = (720, 1280, 3)  # (height, width, channels)


class LatestFrameVideoTrack(VideoStreamTrack):
    """Streams whatever CameraManager's current raw frame is, at the
    track's own self-paced cadence. See module docstring for the
    latest-frame / non-blocking contract."""

    kind = "video"

    def __init__(self, camera_manager, consumer_id: str) -> None:
        super().__init__()
        self._camera_manager = camera_manager
        self._consumer_id = consumer_id
        self._last_frame: Optional[np.ndarray] = None

    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()

        # Non-blocking: never wait on the camera here. If nothing new
        # has landed since our last pull, self._last_frame (the
        # previous frame) is reused, per the module's latest-frame /
        # never-block contract.
        frame_obj = self._camera_manager.get_latest_raw_frame(self._consumer_id)
        if frame_obj is not None:
            self._last_frame = frame_obj.image

        image = self._last_frame
        if image is None:
            image = np.zeros(_BLANK_FRAME_SIZE, dtype=np.uint8)

        video_frame = VideoFrame.from_ndarray(image, format="bgr24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame

    def stop(self) -> None:
        super().stop()
        self._camera_manager.forget_video_consumer(self._consumer_id)


def build_ice_servers() -> List[RTCIceServer]:
    """
    STUN/TURN is opt-in via the WEBRTC_ICE_URLS env var (comma-separated,
    e.g. "stun:stun.l.google.com:19302" or a TURN URL with
    WEBRTC_ICE_USERNAME / WEBRTC_ICE_CREDENTIAL). Default is empty --
    plain host candidates only, which is all a same-LAN Jetson
    deployment needs, and avoids making camera streaming depend on
    internet/STUN reachability for a purely local setup.
    """
    raw = os.getenv("WEBRTC_ICE_URLS", "").strip()
    if not raw:
        return []
    urls = [u.strip() for u in raw.split(",") if u.strip()]
    username = os.getenv("WEBRTC_ICE_USERNAME") or None
    credential = os.getenv("WEBRTC_ICE_CREDENTIAL") or None
    return [RTCIceServer(urls=urls, username=username, credential=credential)]


def build_rtc_configuration() -> RTCConfiguration:
    return RTCConfiguration(iceServers=build_ice_servers())


async def wait_for_ice_gathering_complete(pc: RTCPeerConnection, timeout: float = 10.0) -> None:
    """
    This signaling model is plain HTTP offer/answer (no trickle ICE
    signaling channel), so every candidate has to be baked into the SDP
    we hand back in one response. Waiting for gathering to finish here
    means the client gets a complete, immediately-usable answer from a
    single HTTP round trip.
    """
    if pc.iceGatheringState == "complete":
        return
    done = asyncio.Event()

    @pc.on("icegatheringstatechange")
    def _on_change() -> None:
        if pc.iceGatheringState == "complete":
            done.set()

    try:
        await asyncio.wait_for(done.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning("ICE gathering did not complete within %.1fs; answering with what we have.", timeout)


class PeerConnectionRegistry:
    """
    Tracks every live RTCPeerConnection so `/webcam/stop` and app
    shutdown can deterministically tear them all down. Without this, a
    stopped camera would leave WebRTC viewers holding a connection that
    just silently freezes on the last frame forever, with nothing
    telling the frontend the stream actually ended.
    """

    def __init__(self) -> None:
        self._pcs: Set[RTCPeerConnection] = set()

    def add(self, pc: RTCPeerConnection) -> None:
        self._pcs.add(pc)

    def discard(self, pc: RTCPeerConnection) -> None:
        self._pcs.discard(pc)

    def __len__(self) -> int:
        return len(self._pcs)

    async def close_all(self) -> None:
        pcs = list(self._pcs)
        self._pcs.clear()
        for pc in pcs:
            try:
                await pc.close()
            except Exception:
                logger.exception("Error closing WebRTC peer connection")


def new_consumer_id() -> str:
    return f"webrtc-{uuid.uuid4().hex[:12]}"
