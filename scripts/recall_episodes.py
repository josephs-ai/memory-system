"""
Ask what happened, in a time window.

    python3 scripts/recall_episodes.py --since yesterday
    python3 scripts/recall_episodes.py --since 7d "phase 4"

The semantic store (memory_items) answers "what is true" and dedupes by
content, which is why asking it "what did we do yesterday" returns whatever
old rows share the most vocabulary. This reads the episodic lane instead:
the window is a filter, and text only narrows within it.
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from memory_db import search_episodes, close_pool

_UNITS = {"h": "hours", "d": "days", "w": "weeks"}


def parse_since(value: str):
    """Accept 'yesterday', 'today', '3d', '12h', '2w', or an ISO date."""
    now = datetime.now(timezone.utc)
    text = (value or "").strip().lower()

    if text in {"today", "day"}:
        return now - timedelta(days=1)
    if text == "yesterday":
        return now - timedelta(days=2)
    if text == "week":
        return now - timedelta(weeks=1)

    match = re.fullmatch(r"(\d+)\s*([hdw])", text)
    if match:
        return now - timedelta(**{_UNITS[match.group(2)]: int(match.group(1))})

    try:
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        raise SystemExit(f"could not read --since {value!r} (try: yesterday, 3d, 2w, 2026-08-01)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="*", help="optional words to narrow within the window")
    parser.add_argument("--since", default="1d")
    parser.add_argument("--until", default=None)
    parser.add_argument("--agent", default=None)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    after = parse_since(args.since)
    before = parse_since(args.until) if args.until else None
    query = " ".join(args.query) or None

    rows = search_episodes(
        after_ts=after,
        before_ts=before,
        query_text=query,
        agent=args.agent,
        limit=args.limit,
    )

    window = f"since {after:%Y-%m-%d %H:%M}" + (f" until {before:%Y-%m-%d %H:%M}" if before else "")
    label = f'"{query}" ' if query else ""
    print(f"{len(rows)} episode(s) {label}{window}")
    for e in rows:
        print(f"  {e['started_at']:%Y-%m-%d %H:%M}  [{e['agent']}]  {e['summary'][:100]}")

    if not rows:
        # Distinguish "nothing happened" from "nothing matched", which are very
        # different answers and otherwise look identical.
        total = len(search_episodes(after_ts=after, before_ts=before, limit=1))
        if query and total:
            print("  (episodes exist in this window; none matched the text)")
        else:
            print("  (no activity recorded in this window)")

    close_pool()


if __name__ == "__main__":
    main()
