"""
consolidation_grouper.py — Groups memory items by time window for progressive summarization.

Part of P2: Progressive Summarization / Memory Consolidation.

Groups active memory items into consolidation buckets based on age and
entity/scope similarity. Older items get grouped into wider time windows.

Resolution levels:
    - daily: items 7-30 days old → grouped by day + scope + entity
    - weekly: items 30-90 days old → grouped by week + scope
    - monthly: items 90+ days old → grouped by month + scope

Design decisions:
    - Only groups items with status='active' and memory_type NOT IN ('summary', 'feature_summary')
    - Items already consolidated (resolution_level IS NOT NULL) are skipped
    - Minimum group size of 3 items required (don't summarize singletons)
    - Groups respect scope boundaries (never merge cross-project items)
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from datetime import datetime, timezone, timedelta

import psycopg

LOGGER = logging.getLogger("openclaw.consolidation_grouper")

DB_DSN = os.environ.get("OPENCLAW_MEMORY_DSN", "dbname=openclaw_memory")

# Age thresholds (in days)
DAILY_MIN_AGE = 7
WEEKLY_MIN_AGE = 30
MONTHLY_MIN_AGE = 90

# Minimum items in a group to warrant consolidation
MIN_GROUP_SIZE = 3

# Memory types that should NOT be consolidated
EXCLUDED_TYPES = {"summary", "feature_summary", "project_summary"}


def _iso_week(dt: datetime) -> str:
    """ISO year-week string like '2026-W19'."""
    iso = dt.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def _iso_month(dt: datetime) -> str:
    """Year-month string like '2026-05'."""
    return dt.strftime("%Y-%m")


def _iso_day(dt: datetime) -> str:
    """Date string like '2026-05-13'."""
    return dt.strftime("%Y-%m-%d")


def fetch_consolidation_candidates(
    *,
    min_age_days: int = DAILY_MIN_AGE,
    conn: psycopg.Connection | None = None,
) -> list[dict]:
    """
    Fetch active memory items older than min_age_days that haven't been
    consolidated yet.
    """
    close_conn = False
    if conn is None:
        conn = psycopg.connect(DB_DSN)
        close_conn = True

    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=min_age_days)

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id, text, memory_type, scope, entity, property,
                    first_seen, last_confirmed, confidence,
                    tags, sensitivity
                FROM memory_items
                WHERE status = 'active'
                  AND resolution_level IS NULL
                  AND memory_type NOT IN ('summary', 'feature_summary', 'project_summary')
                  AND (first_seen IS NOT NULL AND first_seen < %s)
                ORDER BY first_seen ASC
                """,
                (cutoff,),
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    finally:
        if close_conn:
            conn.close()


def group_for_consolidation(
    items: list[dict],
    *,
    now: datetime | None = None,
) -> list[dict]:
    """
    Group items into consolidation buckets by resolution level.

    Returns a list of group dicts:
        {
            "resolution": "daily" | "weekly" | "monthly",
            "group_key": str,        # e.g. "2026-05-01|global|browser"
            "scope": str,
            "time_label": str,       # human-readable time window
            "time_range_start": datetime,
            "time_range_end": datetime,
            "items": [dict],
        }
    """
    now = now or datetime.now(timezone.utc)
    groups: dict[str, dict] = {}

    for item in items:
        first_seen = item.get("first_seen")
        if first_seen is None:
            continue

        if first_seen.tzinfo is None:
            first_seen = first_seen.replace(tzinfo=timezone.utc)

        age_days = (now - first_seen).total_seconds() / 86400.0
        scope = item.get("scope") or "global"
        entity = item.get("entity") or ""

        # Determine resolution level based on age
        if age_days >= MONTHLY_MIN_AGE:
            resolution = "monthly"
            time_key = _iso_month(first_seen)
        elif age_days >= WEEKLY_MIN_AGE:
            resolution = "weekly"
            time_key = _iso_week(first_seen)
        elif age_days >= DAILY_MIN_AGE:
            resolution = "daily"
            time_key = _iso_day(first_seen)
        else:
            continue  # Too recent

        # Group key: resolution + time + scope (+ entity for daily)
        if resolution == "daily":
            group_key = f"{resolution}|{time_key}|{scope}|{entity}"
        else:
            group_key = f"{resolution}|{time_key}|{scope}"

        if group_key not in groups:
            groups[group_key] = {
                "resolution": resolution,
                "group_key": group_key,
                "scope": scope,
                "time_label": time_key,
                "items": [],
            }

        groups[group_key]["items"].append(item)

    # Filter out groups below minimum size
    result = []
    for g in groups.values():
        if len(g["items"]) < MIN_GROUP_SIZE:
            continue

        # Compute time range from actual items
        timestamps = [
            i["first_seen"].replace(tzinfo=timezone.utc) if i["first_seen"].tzinfo is None else i["first_seen"]
            for i in g["items"]
            if i.get("first_seen")
        ]
        if timestamps:
            g["time_range_start"] = min(timestamps)
            g["time_range_end"] = max(timestamps)

        result.append(g)

    # Sort by time range start
    result.sort(key=lambda x: x.get("time_range_start") or datetime.min.replace(tzinfo=timezone.utc))

    return result


def consolidation_plan(
    *,
    conn: psycopg.Connection | None = None,
) -> dict:
    """
    Generate a consolidation plan showing what would be consolidated.

    Returns a plan dict with group counts and item counts per resolution level.
    """
    items = fetch_consolidation_candidates(conn=conn)
    groups = group_for_consolidation(items)

    by_resolution = defaultdict(lambda: {"groups": 0, "items": 0})
    for g in groups:
        res = g["resolution"]
        by_resolution[res]["groups"] += 1
        by_resolution[res]["items"] += len(g["items"])

    return {
        "type": "consolidation_plan",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_candidates": len(items),
        "total_groups": len(groups),
        "total_items_to_consolidate": sum(len(g["items"]) for g in groups),
        "by_resolution": dict(by_resolution),
        "groups": [
            {
                "group_key": g["group_key"],
                "resolution": g["resolution"],
                "scope": g["scope"],
                "time_label": g["time_label"],
                "item_count": len(g["items"]),
                "time_range": f"{g.get('time_range_start', '?')} → {g.get('time_range_end', '?')}",
            }
            for g in groups
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Memory consolidation grouper")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("plan", help="Show consolidation plan")
    sub.add_parser("candidates", help="Count consolidation candidates")

    args = parser.parse_args()

    if args.command == "plan":
        plan = consolidation_plan()
        print(json.dumps(plan, indent=2, default=str))

    elif args.command == "candidates":
        items = fetch_consolidation_candidates()
        print(f"Candidates: {len(items)}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
