"""
Heartbeat subsystem: tuning management.

Key functions: compute_effective_interval
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from checkpoint_db import list_recent_triggers, list_recent_checkpoints


def _safe_duration_seconds(row: dict[str, Any]) -> float | None:
    start = row.get("started_at")
    end = row.get("finished_at")
    if not start or not end:
        return None
    try:
        dur = round((end - start).total_seconds(), 3)
        if dur < 0:
            return None
        return dur
    except Exception:
        return None

def compute_effective_interval(default_interval: int = 30) -> dict[str, Any]:
    """
    Priority chain:
    1. failure brake
    2. slow-pipeline cooling
    3. activity settle filter
    4. backlog sprint
    5. steady pulse
    """
    triggers = list_recent_triggers(10)
    checkpoints = list_recent_checkpoints(5)

    pending = sum(1 for t in triggers if t.get("status") == "pending")
    recent_failed = sum(1 for c in checkpoints if c.get("status") == "failed")
    recent_trigger_count = sum(
        1 for t in triggers if t.get("status") in {"pending", "processing", "done"}
    )

    durations: list[float] = []
    for c in checkpoints:
        if c.get("status") != "committed":
            continue
        dur = _safe_duration_seconds(c)
        if dur is not None:
            durations.append(dur)

    avg_duration = round(sum(durations) / len(durations), 3) if durations else None

    if recent_failed >= 3:
        interval = 300
        reason = "failure_emergency_brake"
    elif avg_duration is not None and avg_duration > 20:
        interval = 90
        reason = "slow_pipeline_cooling_period"
    elif recent_trigger_count >= 3:
        interval = 60
        reason = "high_activity_settle_filter"
    elif pending > 0:
        interval = 15
        reason = "healthy_backlog_sprint"
    else:
        interval = int(default_interval)
        reason = "steady_pulse_idle"

    return {
        "effective_interval_seconds": int(interval),
        "reason": reason,
        "snapshot": {
            "pending_triggers": pending,
            "recent_failed_checkpoints": recent_failed,
            "recent_trigger_count": recent_trigger_count,
            "avg_committed_duration_seconds": avg_duration,
            "default_interval_seconds": int(default_interval),
        },
    }
