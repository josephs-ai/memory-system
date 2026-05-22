"""
temporal_scoring.py — Time-aware scoring for memory retrieval.

Part of P3: Temporal Reasoning.

Provides:
1. Time-decay scoring: recent memories ranked higher unless explicitly searching old
2. Temporal query detection: identifies "when"-oriented queries
3. Temporal range filtering: extract date ranges from queries
4. Time-aware boost/penalty: applied during retrieval scoring

Design decisions:
- Decay is logarithmic, not exponential (gentle degradation, not cliff)
- Temporal queries REVERSE the decay (prefer older items matching the time reference)
- No LLM calls — deterministic regex + scoring math
- Graceful degradation: returns neutral score on error
"""

from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timezone, timedelta

LOGGER = logging.getLogger("openclaw.temporal_scoring")

# Scoring constants
RECENCY_WEIGHT = 0.10          # Max boost for very recent items
RECENCY_HALF_LIFE_DAYS = 14   # Days until recency boost halves
STALENESS_PENALTY = -0.08     # Max penalty for very old items
STALENESS_ONSET_DAYS = 60     # Days before staleness penalty kicks in
TEMPORAL_QUERY_BOOST = 0.12   # Boost for items matching temporal query's time reference


# ---------------------------------------------------------------------------
# Temporal query detection
# ---------------------------------------------------------------------------

_TEMPORAL_KEYWORDS = re.compile(
    r'\b('
    r'when|yesterday|today|last\s+week|last\s+month|last\s+year|'
    r'ago|recently|earlier|before|after|since|until|during|'
    r'timeline|history|chronolog|sequence|changed|happened|'
    r'january|february|march|april|may|june|july|august|'
    r'september|october|november|december|'
    r'monday|tuesday|wednesday|thursday|friday|saturday|sunday'
    r')\b',
    re.IGNORECASE,
)

_DATE_PATTERNS = [
    # ISO dates: 2026-05-13
    re.compile(r'(\d{4})-(\d{2})-(\d{2})'),
    # Relative: "N days/weeks/months ago"
    re.compile(r'(\d+)\s+(days?|weeks?|months?)\s+ago', re.IGNORECASE),
    # "last N days/weeks"
    re.compile(r'last\s+(\d+)\s+(days?|weeks?|months?)', re.IGNORECASE),
    # Named months: "in March", "March 2026"
    re.compile(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s*(\d{4})?\b', re.IGNORECASE),
]

MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def is_temporal_query(query: str) -> bool:
    """Detect if a query is primarily about when something happened."""
    if not query:
        return False
    hits = len(_TEMPORAL_KEYWORDS.findall(query))
    words = max(len(query.split()), 1)

    # Also check for date patterns (ISO dates, "N days ago", etc.)
    has_date_pattern = any(p.search(query) for p in _DATE_PATTERNS)
    if has_date_pattern:
        hits += 1  # Date pattern counts as a temporal signal

    # Temporal if ≥2 hits or (≥1 hit and density > 12%)
    return hits >= 2 or (hits >= 1 and hits / words > 0.12)


def extract_time_reference(query: str) -> dict | None:
    """
    Extract a time reference from a query.

    Returns:
        {
            "type": "absolute" | "relative" | "named_month",
            "start": datetime,
            "end": datetime,
        }
        or None if no time reference found.
    """
    if not query:
        return None

    now = datetime.now(timezone.utc)

    # ISO date
    m = _DATE_PATTERNS[0].search(query)
    if m:
        try:
            dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=timezone.utc)
            return {
                "type": "absolute",
                "start": dt,
                "end": dt + timedelta(days=1),
            }
        except ValueError:
            pass

    # "N days/weeks/months ago"
    m = _DATE_PATTERNS[1].search(query)
    if m:
        n = int(m.group(1))
        unit = m.group(2).lower().rstrip("s")
        if unit == "day":
            delta = timedelta(days=n)
        elif unit == "week":
            delta = timedelta(weeks=n)
        elif unit == "month":
            delta = timedelta(days=n * 30)
        else:
            delta = timedelta(days=n)
        ref = now - delta
        return {
            "type": "relative",
            "start": ref - timedelta(days=1),
            "end": ref + timedelta(days=1),
        }

    # "last N days/weeks"
    m = _DATE_PATTERNS[2].search(query)
    if m:
        n = int(m.group(1))
        unit = m.group(2).lower().rstrip("s")
        if unit == "day":
            delta = timedelta(days=n)
        elif unit == "week":
            delta = timedelta(weeks=n)
        elif unit == "month":
            delta = timedelta(days=n * 30)
        else:
            delta = timedelta(days=n)
        return {
            "type": "relative",
            "start": now - delta,
            "end": now,
        }

    # Named month
    m = _DATE_PATTERNS[3].search(query)
    if m:
        month_name = m.group(1).lower()
        year = int(m.group(2)) if m.group(2) else now.year
        month = MONTH_MAP.get(month_name)
        if month:
            start = datetime(year, month, 1, tzinfo=timezone.utc)
            if month == 12:
                end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
            else:
                end = datetime(year, month + 1, 1, tzinfo=timezone.utc)
            return {
                "type": "named_month",
                "start": start,
                "end": end,
            }

    return None


# ---------------------------------------------------------------------------
# Time-decay scoring
# ---------------------------------------------------------------------------

def recency_score(item_timestamp: datetime | None, now: datetime | None = None) -> float:
    """
    Compute a recency-based score modifier.

    Recent items get a positive boost, old items get a penalty.
    Items in the middle (< STALENESS_ONSET_DAYS) get a gentle decay.

    Returns: float in [STALENESS_PENALTY, RECENCY_WEIGHT]
    """
    if item_timestamp is None:
        return 0.0

    now = now or datetime.now(timezone.utc)
    if item_timestamp.tzinfo is None:
        item_timestamp = item_timestamp.replace(tzinfo=timezone.utc)

    age_days = max((now - item_timestamp).total_seconds() / 86400.0, 0.0)

    if age_days <= 1:
        return RECENCY_WEIGHT

    # Logarithmic decay from RECENCY_WEIGHT toward 0
    decay = RECENCY_WEIGHT * (1.0 / (1.0 + math.log2(1.0 + age_days / RECENCY_HALF_LIFE_DAYS)))

    # Staleness penalty for very old items
    if age_days > STALENESS_ONSET_DAYS:
        staleness_factor = min((age_days - STALENESS_ONSET_DAYS) / 180.0, 1.0)
        decay += STALENESS_PENALTY * staleness_factor

    return max(STALENESS_PENALTY, min(RECENCY_WEIGHT, decay))


def temporal_match_score(
    item_timestamp: datetime | None,
    time_ref: dict | None,
) -> float:
    """
    Compute a temporal match boost when item falls within a query's time reference.

    Returns TEMPORAL_QUERY_BOOST if item timestamp is within the reference window,
    0.0 otherwise.
    """
    if item_timestamp is None or time_ref is None:
        return 0.0

    if item_timestamp.tzinfo is None:
        item_timestamp = item_timestamp.replace(tzinfo=timezone.utc)

    start = time_ref.get("start")
    end = time_ref.get("end")

    if start and end and start <= item_timestamp <= end:
        return TEMPORAL_QUERY_BOOST

    return 0.0


def compute_temporal_boost(
    item_timestamp: datetime | None,
    query: str,
    *,
    now: datetime | None = None,
) -> float:
    """
    Compute the total temporal score modifier for an item.

    For temporal queries: time-reference match boost (no recency decay).
    For non-temporal queries: recency decay (recent = boost, old = penalty).
    """
    try:
        if is_temporal_query(query):
            time_ref = extract_time_reference(query)
            if time_ref:
                return temporal_match_score(item_timestamp, time_ref)
            # Temporal query but no extractable reference → no boost
            return 0.0
        else:
            return recency_score(item_timestamp, now=now)
    except Exception:
        LOGGER.debug("temporal scoring failed", exc_info=True)
        return 0.0


def compute_temporal_boost_batch(
    items: list[dict],
    query: str,
    *,
    timestamp_key: str = "first_seen",
    now: datetime | None = None,
) -> dict[str, float]:
    """
    Batch compute temporal boosts for multiple items.

    Returns dict mapping item_id → temporal boost.
    """
    result = {}
    is_temporal = is_temporal_query(query)
    time_ref = extract_time_reference(query) if is_temporal else None
    now = now or datetime.now(timezone.utc)

    for item in items:
        item_id = item.get("id") or item.get("item_id")
        if not item_id:
            continue

        ts = item.get(timestamp_key)
        if is_temporal and time_ref:
            result[item_id] = temporal_match_score(ts, time_ref)
        elif not is_temporal:
            result[item_id] = recency_score(ts, now=now)
        else:
            result[item_id] = 0.0

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Temporal scoring tools")
    sub = parser.add_subparsers(dest="command")

    detect_p = sub.add_parser("detect", help="Detect if query is temporal")
    detect_p.add_argument("query")

    extract_p = sub.add_parser("extract", help="Extract time reference from query")
    extract_p.add_argument("query")

    score_p = sub.add_parser("score", help="Compute recency score for an age")
    score_p.add_argument("age_days", type=float)

    args = parser.parse_args()

    if args.command == "detect":
        result = is_temporal_query(args.query)
        print(json.dumps({"query": args.query, "is_temporal": result}))

    elif args.command == "extract":
        ref = extract_time_reference(args.query)
        print(json.dumps({"query": args.query, "time_reference": ref}, default=str))

    elif args.command == "score":
        now = datetime.now(timezone.utc)
        ts = now - timedelta(days=args.age_days)
        score = recency_score(ts, now=now)
        print(json.dumps({"age_days": args.age_days, "recency_score": score}))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
