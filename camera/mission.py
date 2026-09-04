"""
Mission recorder.

Converts a live tracking session into a structured mission object:
mission metadata + a meaningful event log + operator-configurable
alerts. It sits ON TOP of CameraManager's existing session state --
it does NOT track, detect, or count anything itself. CameraManager
already produces the gated unique-track set and the per-class counts
(from detector.names); MissionRecorder just:

  - times the mission (begin / end / duration),
  - logs meaningful events (a class seen for the first time, an alert
    firing),
  - evaluates operator alert rules once per frame with a per-rule
    LATCH so an alert fires once per threshold crossing, never once
    per frame.

Nothing here is model-specific. Every class name comes from the count
dicts CameraManager passes in.
"""

from __future__ import annotations

import datetime
import time
import uuid
from typing import Optional


def format_offset(seconds: float) -> str:
    """Mission-relative time as HH:MM:SS."""
    total = max(0, int(round(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def utc_iso() -> str:
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"


_VALID_METRICS = ("visible", "unique")
_SEVERITY_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


class AlertEngine:
    """
    Evaluates a flat list of operator alert rules once per frame.

    Rule shape (model-class-agnostic):
        {
          "class":     "<any model class name>",
          "metric":    "visible" | "unique",   # default "unique"
          "operator":  ">=",                   # only ">=" supported
          "threshold": <int>,
          "severity":  "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
          "enabled":   true
        }

    LATCH: a rule fires ONCE when its condition first becomes true and
    does not fire again until the value has dropped back below the
    threshold (re-arm). This is what prevents hundreds of identical
    alerts while a count sits at, e.g., 3.
    """

    def __init__(self, rules: Optional[list] = None) -> None:
        self._rules: list = []
        for raw in rules or []:
            cls = raw.get("class")
            if not cls or not raw.get("enabled", True):
                continue
            metric = raw.get("metric", "unique")
            if metric not in _VALID_METRICS:
                metric = "unique"
            severity = str(raw.get("severity", "MEDIUM")).upper()
            if severity not in _SEVERITY_ORDER:
                severity = "MEDIUM"
            try:
                threshold = int(raw.get("threshold", 1))
            except (TypeError, ValueError):
                continue
            if threshold < 1:
                threshold = 1
            self._rules.append(
                {
                    "class": str(cls),
                    "metric": metric,
                    "operator": ">=",
                    "threshold": threshold,
                    "severity": severity,
                }
            )
        self._fired = [False] * len(self._rules)

    @property
    def rule_count(self) -> int:
        return len(self._rules)

    def evaluate(
        self,
        *,
        visible_counts: dict,
        unique_counts: dict,
        offset_s: float,
        timestamp_iso: str,
    ) -> list:
        triggered = []
        for i, rule in enumerate(self._rules):
            source = visible_counts if rule["metric"] == "visible" else unique_counts
            actual = int(source.get(rule["class"], 0))
            meets = actual >= rule["threshold"]

            if meets and not self._fired[i]:
                self._fired[i] = True
                metric_word = "visible count" if rule["metric"] == "visible" else "unique count"
                triggered.append(
                    {
                        "alert_id": uuid.uuid4().hex[:12],
                        "timestamp": timestamp_iso,
                        "offset": format_offset(offset_s),
                        "class": rule["class"],
                        "metric": rule["metric"],
                        "condition": f"{rule['metric']}_count >= {rule['threshold']}",
                        "threshold": rule["threshold"],
                        "actual_value": actual,
                        "severity": rule["severity"],
                        "message": (
                            f"{rule['class']} {metric_word} reached {actual} "
                            f"(threshold {rule['threshold']})"
                        ),
                    }
                )
            elif not meets and self._fired[i]:
                self._fired[i] = False  # re-arm

        return triggered


class MissionRecorder:
    """
    One mission = one begin()/end() cycle. observe() is called once per
    processed frame by CameraManager's inference loop, under
    CameraManager._lock (so no separate lock is needed here).
    """

    _MAX_EVENTS = 500
    _MAX_ALERTS = 500

    def __init__(self) -> None:
        self._active = False
        self._session_id: Optional[str] = None
        self._start_iso: Optional[str] = None
        self._end_iso: Optional[str] = None
        self._start_monotonic = 0.0
        self._duration_s = 0.0

        self._source_type = "Webcam"
        self._source_name = "Webcam"
        self._resolution: Optional[str] = None
        self._recording_enabled = False
        # Set by CameraManager._finalize_recording() on stop() once a
        # usable mp4 exists. Relative URL, resolved by the frontend.
        self._video_url: Optional[str] = None

        self._engine = AlertEngine([])
        self._events: list = []
        self._alerts: list = []
        self._seen_classes: set = set()

    # ------------------------------------------------------------------
    def begin(
        self,
        *,
        session_id: str,
        source_type: str,
        source_name: str,
        resolution: Optional[str],
        recording_enabled: bool,
        alert_rules: Optional[list],
    ) -> None:
        self._active = True
        self._session_id = session_id
        self._start_iso = utc_iso()
        self._end_iso = None
        self._start_monotonic = time.monotonic()
        self._duration_s = 0.0

        self._source_type = source_type or "Webcam"
        self._source_name = source_name or source_type or "Webcam"
        self._resolution = resolution
        self._recording_enabled = bool(recording_enabled)
        self._video_url = None

        self._engine = AlertEngine(alert_rules)
        self._events = []
        self._alerts = []
        self._seen_classes = set()

    def end(self) -> None:
        if not self._active:
            return
        self._end_iso = utc_iso()
        self._duration_s = max(0.0, time.monotonic() - self._start_monotonic)
        self._active = False

    def clear(self) -> None:
        self.__init__()

    def set_resolution(self, resolution: Optional[str]) -> None:
        if resolution:
            self._resolution = resolution

    def set_video(self, url: Optional[str]) -> None:
        """CameraManager calls this on stop() once the mission mp4 is
        finalized. `url` is a relative path the frontend resolves
        against the detection API base."""
        self._video_url = url

    @property
    def offset_now(self) -> float:
        return max(0.0, time.monotonic() - self._start_monotonic) if self._active else 0.0

    # ------------------------------------------------------------------
    def observe(self, *, visible_class_counts: dict, unique_class_counts: dict) -> None:
        """Per processed frame. `visible_class_counts` = classes on
        screen right now; `unique_class_counts` = SESSION unique
        (gated) per class. Both come from CameraManager, computed from
        detector.names."""
        if not self._active:
            return

        offset = max(0.0, time.monotonic() - self._start_monotonic)
        now_iso = utc_iso()
        self._duration_s = offset

        for cls, count in unique_class_counts.items():
            if count > 0 and cls not in self._seen_classes:
                self._seen_classes.add(cls)
                self._append_event(
                    {
                        "event_id": uuid.uuid4().hex[:12],
                        "timestamp": now_iso,
                        "offset": format_offset(offset),
                        "type": "class_first_seen",
                        "class": cls,
                        "message": f"{cls} first observed",
                    }
                )

        for alert in self._engine.evaluate(
            visible_counts=visible_class_counts,
            unique_counts=unique_class_counts,
            offset_s=offset,
            timestamp_iso=now_iso,
        ):
            self._append_alert(alert)
            self._append_event(
                {
                    "event_id": uuid.uuid4().hex[:12],
                    "timestamp": alert["timestamp"],
                    "offset": alert["offset"],
                    "type": "alert_triggered",
                    "class": alert["class"],
                    "severity": alert["severity"],
                    "message": alert["message"],
                }
            )

    def _append_event(self, event: dict) -> None:
        self._events.append(event)
        if len(self._events) > self._MAX_EVENTS:
            self._events = self._events[-self._MAX_EVENTS:]

    def _append_alert(self, alert: dict) -> None:
        self._alerts.append(alert)
        if len(self._alerts) > self._MAX_ALERTS:
            self._alerts = self._alerts[-self._MAX_ALERTS:]

    # ------------------------------------------------------------------
    def to_dict(self, *, video_url: Optional[str] = None) -> dict:
        vu = video_url if video_url is not None else self._video_url
        highest = None
        if self._alerts:
            highest = max(
                (a["severity"] for a in self._alerts),
                key=lambda s: _SEVERITY_ORDER.get(s, 0),
            )
        return {
            # Human mission id (MIS-YYYYMMDD-NNN) is assigned by the Node
            # backend on save; the FastAPI side only has the session id.
            "mission_id": None,
            "session_id": self._session_id,
            "start_time": self._start_iso,
            "end_time": self._end_iso,
            "duration_seconds": round(self._duration_s, 1),
            "duration_hms": format_offset(self._duration_s),
            "active": self._active,
            "source_type": self._source_type,
            "source_name": self._source_name,
            "resolution": self._resolution,
            "recording": {
                "enabled": self._recording_enabled,
                "video_available": bool(vu),
                "video_url": vu,
            },
            "alert_rule_count": self._engine.rule_count,
            "events": list(self._events),
            "alerts": list(self._alerts),
            "alert_count": len(self._alerts),
            "highest_severity": highest,
        }
