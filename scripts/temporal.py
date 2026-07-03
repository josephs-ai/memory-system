"""
temporal.py — single source of truth for memory temporal semantics.

Phase T0 foundation for temporal-aware retrieval. Centralizes:
  - canonical effective-timestamp resolution (COALESCE order)
  - age / bucket / human-label computation
  - temporal confidence flagging
  - absolute recency-decay scoring (used by ranking, Phase T2)

Design invariants:
  - Deterministic: callers inject `as_of` (no hidden now()) so tests are stable.
  - Additive: enrichment never mutates existing keys, only adds new ones.
  - Honest: missing/garbage timestamps are flagged low-confidence and the
    recency-decay term is ZEROED (never silently floored to epoch in ranking).

Canonical resolution order:
    effective_ts = COALESCE(last_confirmed, first_seen, created_at)
Rationale: last_confirmed reflects the most recent time the system reaffirmed
the fact; first_seen is when it first appeared; created_at is row-insert time.
We prefer the strongest signal of "when this memory was true/known".
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

# Order matters: strongest temporal signal first.
EFFECTIVE_TS_KEYS = ("last_confirmed", "first_seen", "created_at")

# Bucket thresholds in days (upper-exclusive). Tunable via config later.
BUCKET_TODAY_DAYS = 1.0
BUCKET_WEEK_DAYS = 7.0
BUCKET_MONTH_DAYS = 31.0

# Absolute recency-decay defaults (Phase T2). Overridable from config.
DEFAULT_RECENCY_WEIGHT = 0.6
DEFAULT_HALF_LIFE_DAYS = 14.0

_PARSE_FORMATS = (
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
)


def _coerce_dt(value) -> datetime | None:
    """Parse a timestamp value (datetime or string) to an aware UTC datetime.

    Returns None if the value is missing or unparseable. Naive datetimes are
    assumed UTC (matches TIMESTAMPTZ storage convention here).
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        s = str(value).strip()
        if not s:
            return None
        # Normalize ISO-ish strings: strip 'T', timezone suffix for strptime.
        s_norm = s.replace("T", " ")
        # Drop timezone offset / Z for the strptime fallbacks.
        for cut in ("+", "Z"):
            if cut in s_norm:
                s_norm = s_norm.split(cut)[0]
        s_norm = s_norm.strip()
        for fmt in _PARSE_FORMATS:
            try:
                return datetime.strptime(s_norm[:26], fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
    except Exception:
        return None
    return None


def parse_ts(value) -> datetime | None:
    """Public timestamp parser — the single source of truth for converting a
    user/API-supplied temporal-filter value (ISO-8601 string, YYYY-MM-DD, or
    datetime) into an aware UTC datetime. Returns None if missing/unparseable.

    Used by T3 time-window filters so the SQL push-down and the Qdrant
    push-down parse `since`/`until`/`as_of` identically.
    """
    return _coerce_dt(value)


def resolve_effective_ts(item: dict) -> tuple[datetime | None, str]:
    """Resolve canonical effective timestamp for a memory item.

    Returns (effective_dt_or_None, confidence) where confidence is:
      - "high": resolved from last_confirmed or first_seen
      - "low" : only created_at was available (row-insert proxy)
      - "none": no usable timestamp at all
    """
    for idx, key in enumerate(EFFECTIVE_TS_KEYS):
        dt = _coerce_dt(item.get(key))
        if dt is not None:
            # last_confirmed/first_seen => high; created_at-only => low
            confidence = "high" if idx < 2 else "low"
            return dt, confidence
    return None, "none"


def _now(as_of: datetime | None) -> datetime:
    if as_of is None:
        return datetime.now(timezone.utc)
    return as_of if as_of.tzinfo else as_of.replace(tzinfo=timezone.utc)


def age_days(effective_dt: datetime | None, as_of: datetime | None = None) -> float | None:
    """Whole+fractional days between effective_dt and as_of. None if undated.

    Negative ages (future timestamps / clock skew) are clamped to 0.0 so they
    can't fabricate an inflated recency bonus.
    """
    if effective_dt is None:
        return None
    delta = _now(as_of) - effective_dt
    days = delta.total_seconds() / 86400.0
    return max(0.0, days)


def temporal_bucket(age: float | None) -> str:
    """Cohort label for grouping. 'undated' when age is unknown."""
    if age is None:
        return "undated"
    if age < BUCKET_TODAY_DAYS:
        return "today"
    if age < BUCKET_WEEK_DAYS:
        return "this_week"
    if age < BUCKET_MONTH_DAYS:
        return "this_month"
    return "older"


def temporal_label(age: float | None) -> str:
    """Human-readable relative label, e.g. '3 days ago'."""
    if age is None:
        return "undated"
    if age < (1.0 / 24.0):
        return "just now"
    if age < 1.0:
        hours = max(1, int(round(age * 24)))
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    if age < 7.0:
        days = int(round(age))
        return f"{days} day{'s' if days != 1 else ''} ago"
    if age < 31.0:
        weeks = int(round(age / 7.0))
        return f"{weeks} week{'s' if weeks != 1 else ''} ago"
    if age < 365.0:
        months = int(round(age / 30.4))
        return f"{months} month{'s' if months != 1 else ''} ago"
    years = round(age / 365.0, 1)
    return f"{years} year{'s' if years != 1.0 else ''} ago"


def recency_bonus(
    age: float | None,
    confidence: str,
    weight: float = DEFAULT_RECENCY_WEIGHT,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
) -> float:
    """Absolute recency-decay score term (Phase T2).

    bonus = weight * exp(-ln(2) * age / half_life)

    Returns 0.0 when undated or low/none confidence — we never let a weak
    timestamp manufacture ranking lift. weight=0 reproduces legacy ordering.
    """
    if age is None or confidence != "high" or weight <= 0.0 or half_life_days <= 0.0:
        return 0.0
    return weight * math.exp(-math.log(2.0) * age / half_life_days)


def enrich_item(
    item: dict,
    as_of: datetime | None = None,
    weight: float = DEFAULT_RECENCY_WEIGHT,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
) -> dict:
    """Augment a result row with agent-facing temporal fields (additive).

    Adds: effective_ts (ISO-8601 | None), age_days, temporal_bucket,
          temporal_label, temporal_confidence, recency_bonus.
    Does not overwrite existing keys.

    weight/half_life_days control the surfaced recency_bonus so it agrees with
    the SQL ranking term (which reads the same config). Defaults preserve prior
    behavior for callers that don't pass them.
    """
    eff_dt, confidence = resolve_effective_ts(item)
    age = age_days(eff_dt, as_of)
    item["effective_ts"] = eff_dt.astimezone(timezone.utc).isoformat() if eff_dt else None
    item["age_days"] = round(age, 2) if age is not None else None
    item["temporal_bucket"] = temporal_bucket(age)
    item["temporal_label"] = temporal_label(age)
    item["temporal_confidence"] = confidence
    item["recency_bonus"] = round(
        recency_bonus(age, confidence, weight=weight, half_life_days=half_life_days), 4
    )
    return item


def enrich_rows(
    rows: list[dict],
    as_of: datetime | None = None,
    weight: float = DEFAULT_RECENCY_WEIGHT,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
) -> list[dict]:
    """Enrich a list of result rows in place; returns the same list."""
    for r in rows:
        enrich_item(r, as_of, weight=weight, half_life_days=half_life_days)
    return rows


# Stable presentation order for cohorts (newest-first, undated last).
COHORT_ORDER = ("today", "this_week", "this_month", "older", "undated")


def temporal_cohorts(rows: list[dict], as_of: datetime | None = None) -> dict:
    """Group already-enriched result rows into temporal cohorts (Phase T4).

    Returns a SUMMARY structure (additive, does not reorder/mutate rows):
      {
        "order": [...non-empty cohort names, newest-first...],
        "counts": {cohort: n, ...},          # only non-empty cohorts
        "buckets": {                          # only non-empty cohorts
            cohort: {
              "count": n,
              "newest_label": "2 days ago" | None,
              "oldest_label": "3 weeks ago" | None,
              "item_indexes": [i, ...]        # indexes into the input `rows`
            }, ...
        }
      }

    Idempotent on enrichment: if a row lacks temporal fields (caller skipped
    enrich), they're computed on the fly so cohorts are always correct. Ranking
    order of `rows` is never changed — this is a read-only view over results.
    """
    buckets: dict[str, dict] = {}
    for idx, row in enumerate(rows):
        bucket = row.get("temporal_bucket")
        age = row.get("age_days")
        label = row.get("temporal_label")
        # Fall back to computing if the row wasn't enriched.
        if bucket is None:
            eff_dt, _conf = resolve_effective_ts(row)
            age = age_days(eff_dt, as_of)
            bucket = temporal_bucket(age)
            label = temporal_label(age)
        b = buckets.setdefault(
            bucket, {"count": 0, "_min_age": None, "_max_age": None,
                     "_newest_label": None, "_oldest_label": None, "item_indexes": []}
        )
        b["count"] += 1
        b["item_indexes"].append(idx)
        # Track newest (min age) and oldest (max age) within the cohort.
        if age is not None:
            if b["_min_age"] is None or age < b["_min_age"]:
                b["_min_age"] = age
                b["_newest_label"] = label
            if b["_max_age"] is None or age > b["_max_age"]:
                b["_max_age"] = age
                b["_oldest_label"] = label

    order = [c for c in COHORT_ORDER if c in buckets]
    counts = {c: buckets[c]["count"] for c in order}
    out_buckets = {
        c: {
            "count": buckets[c]["count"],
            "newest_label": buckets[c]["_newest_label"],
            "oldest_label": buckets[c]["_oldest_label"],
            "item_indexes": buckets[c]["item_indexes"],
        }
        for c in order
    }
    return {"order": order, "counts": counts, "buckets": out_buckets}


def temporal_signals(rows: list[dict]) -> dict:
    """Compact observability summary of a result set's temporal profile (T5).

    Designed for logging / metrics: small, JSON-safe, no PII. Reads the
    already-enriched fields (age_days, recency_bonus, temporal_confidence) and
    falls back to computing age if missing. Returns:
      {
        "n": total rows,
        "dated": rows with a usable timestamp,
        "undated": rows with none,
        "age_days": {"min","max","mean","median"} | null (over dated rows),
        "recency_bonus": {"min","max","mean"} | null,
        "confidence": {"high","low","none"}   # counts
      }
    """
    n = len(rows)
    ages: list[float] = []
    bonuses: list[float] = []
    conf = {"high": 0, "low": 0, "none": 0}

    for row in rows:
        c = row.get("temporal_confidence")
        if c not in conf:
            # not enriched — resolve confidence + age on the fly
            eff_dt, c = resolve_effective_ts(row)
            age = age_days(eff_dt, None)
        else:
            age = row.get("age_days")
        conf[c] = conf.get(c, 0) + 1
        if age is not None:
            ages.append(float(age))
        rb = row.get("recency_bonus")
        if isinstance(rb, (int, float)):
            bonuses.append(float(rb))

    def _stats(vals: list[float], want_median: bool) -> dict | None:
        if not vals:
            return None
        s = sorted(vals)
        out = {
            "min": round(s[0], 3),
            "max": round(s[-1], 3),
            "mean": round(sum(s) / len(s), 3),
        }
        if want_median:
            mid = len(s) // 2
            out["median"] = round(s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2, 3)
        return out

    return {
        "n": n,
        "dated": len(ages),
        "undated": n - len(ages),
        "age_days": _stats(ages, want_median=True),
        "recency_bonus": _stats(bonuses, want_median=False),
        "confidence": conf,
    }
