"""
Generate human-readable timeline views from memory event ledger data.
Produces daily markdown summaries for chronological review.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

WORKSPACE = Path.home() / ".openclaw" / "workspace"
TIMELINE_ROOT = WORKSPACE / ".memory-index" / "timeline"
DEFAULT_LEDGER = TIMELINE_ROOT / "events.jsonl"
DAILY_DIR = TIMELINE_ROOT / "daily"
LATEST_SUMMARY = TIMELINE_ROOT / "latest.md"


def load_ledger(path: Path = DEFAULT_LEDGER) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    rows.sort(key=lambda row: (str(row.get("day_key") or ""), str(row.get("timestamp") or ""), str(row.get("event_id") or "")))
    return rows



def _bullet_for_event(event: dict[str, Any]) -> str:
    role = str(event.get("role") or "unknown").upper()
    text = " ".join(str(event.get("text") or "").split())
    timestamp = str(event.get("timestamp") or "")
    hhmm = timestamp[11:16] if len(timestamp) >= 16 else ""
    prefix = f"[{hhmm}] " if hhmm else ""
    return f"- {prefix}{role}: {text}"



def render_daily_views(events: list[dict[str, Any]], *, daily_dir: Path = DAILY_DIR, latest_summary_path: Path = LATEST_SUMMARY) -> dict[str, Any]:
    daily_dir.mkdir(parents=True, exist_ok=True)
    latest_summary_path.parent.mkdir(parents=True, exist_ok=True)

    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        day_key = str(event.get("day_key") or "").strip()
        if not day_key:
            continue
        by_day[day_key].append(event)

    written_days: list[str] = []
    for day_key, rows in sorted(by_day.items()):
        lines = [f"# Timeline summary for {day_key}", "", "## Events", ""]
        seen: set[str] = set()
        for row in rows:
            bullet = _bullet_for_event(row)
            if bullet in seen:
                continue
            seen.add(bullet)
            lines.append(bullet)
        lines.append("")
        (daily_dir / f"{day_key}.md").write_text("\n".join(lines), encoding="utf-8")
        written_days.append(day_key)

    latest_lines = ["# Latest timeline summary", ""]
    for row in events[-20:]:
        latest_lines.append(_bullet_for_event(row))
    latest_lines.append("")
    latest_summary_path.write_text("\n".join(latest_lines), encoding="utf-8")

    return {
        "days_rendered": len(written_days),
        "latest_day": written_days[-1] if written_days else None,
        "latest_summary": str(latest_summary_path),
    }



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER))
    parser.add_argument("--daily-dir", default=str(DAILY_DIR))
    parser.add_argument("--latest-summary", default=str(LATEST_SUMMARY))
    args = parser.parse_args()

    events = load_ledger(Path(args.ledger))
    stats = render_daily_views(events, daily_dir=Path(args.daily_dir), latest_summary_path=Path(args.latest_summary))
    print(json.dumps({"ok": True, **stats}))


if __name__ == "__main__":
    main()
