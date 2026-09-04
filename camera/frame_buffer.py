"""
Latest-frame-only buffer.

This is the core latency primitive of the whole pipeline: it never holds
more than one frame. A new frame always overwrites whatever is waiting,
so a consumer that is momentarily slower than the producer (e.g. YOLO
inference slower than camera FPS) always gets the newest available frame
instead of working through a backlog.

There is deliberately no queue.Queue(maxsize=N) with N>1 anywhere in this
codebase for frame data -- that would reintroduce the staleness bug.

Multi-consumer support
-----------------------
Originally this buffer had exactly one reader (the inference worker).
The WebRTC video track is a second, independent reader pulling from the
same GStreamer appsink output (raw frames, before detection boxes are
known) at its own cadence. A single shared "have you seen the latest
frame" flag can't serve two readers -- whichever one calls get_latest()
first would mark the frame seen and starve the other.

So "seen" is now tracked per `consumer_id` in a dict, and every reader
still independently gets latest-frame (never-queued, never-backlogged)
semantics relative to its *own* last read. The `dropped_frames` /
`total_frames` stats in `/webcam/status` stay tied to the single
"primary" consumer (the inference worker) so that number keeps meaning
what it always meant: frames the detector couldn't keep up with. Other
consumers (WebRTC viewers) don't affect that counter -- them "missing" a
frame because they weren't ready to pull yet isn't a real drop, since
they just repeat the previous frame instead of falling behind.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

# Reserved consumer_id used by every call site that doesn't pass one
# explicitly (i.e. the inference worker), so existing behavior/stats are
# unchanged for anyone not opting into multi-consumer reads.
PRIMARY_CONSUMER = "_primary"


@dataclass
class Frame:
    image: np.ndarray          # BGR, OpenCV-compatible
    timestamp: float           # time.monotonic() when captured
    frame_id: int


class LatestFrameBuffer:
    """
    Single-slot, thread-safe "mailbox" for frames, readable by multiple
    independent consumers.

    - put(): always overwrites the previous frame (drop old frame).
    - get_latest(): returns the newest frame this `consumer_id` hasn't
      seen yet, or None if nothing new has arrived since that consumer's
      last read (when block=False), or blocks up to `timeout` seconds
      waiting for a fresh one (when block=True).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._frame: Optional[Frame] = None
        self._seen_ids: Dict[str, int] = {}
        self._next_id: int = 0
        self._dropped_count: int = 0
        self._total_count: int = 0

    def put(self, image: np.ndarray) -> None:
        with self._cond:
            primary_seen = self._seen_ids.get(PRIMARY_CONSUMER, -1)
            if self._frame is not None and self._frame.frame_id != primary_seen:
                # Previous frame was never consumed by the primary
                # (inference) consumer -> it's being dropped.
                self._dropped_count += 1
            self._frame = Frame(image=image, timestamp=time.monotonic(), frame_id=self._next_id)
            self._next_id += 1
            self._total_count += 1
            self._cond.notify_all()

    def get_latest(
        self,
        block: bool = True,
        timeout: Optional[float] = 1.0,
        consumer_id: str = PRIMARY_CONSUMER,
    ) -> Optional[Frame]:
        with self._cond:
            seen = self._seen_ids.get(consumer_id, -1)
            if block:
                deadline = None if timeout is None else time.monotonic() + timeout
                while self._frame is None or self._frame.frame_id == seen:
                    remaining = None if deadline is None else deadline - time.monotonic()
                    if remaining is not None and remaining <= 0:
                        return None
                    self._cond.wait(timeout=remaining)
                    seen = self._seen_ids.get(consumer_id, -1)
            if self._frame is None or self._frame.frame_id == seen:
                return None
            self._seen_ids[consumer_id] = self._frame.frame_id
            return self._frame

    def forget_consumer(self, consumer_id: str) -> None:
        """Drop the seen-id bookkeeping for a consumer that's gone away
        (e.g. a WebRTC viewer that disconnected) so the dict doesn't grow
        without bound across many connect/disconnect cycles."""
        with self._cond:
            self._seen_ids.pop(consumer_id, None)

    def clear(self) -> None:
        with self._cond:
            self._frame = None
            self._seen_ids = {}
            self._dropped_count = 0
            self._total_count = 0
            self._next_id = 0

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "dropped_frames": self._dropped_count,
                "total_frames": self._total_count,
            }
