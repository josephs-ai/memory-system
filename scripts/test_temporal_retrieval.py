"""
Tests for temporal-aware retrieval foundation (Phase T0 + T1).

Covers the temporal.py contract: effective-ts resolution, confidence flagging,
age/bucket/label computation, absolute recency decay, and additive enrichment.
All time-dependent assertions use a frozen `as_of` for determinism.
"""
from datetime import datetime, timezone, timedelta

import pytest

import temporal as T

AS_OF = datetime(2026, 6, 18, 21, 50, tzinfo=timezone.utc)


def _enrich(**kw):
    return T.enrich_item(dict(kw), AS_OF)


# --- effective-ts resolution + confidence ---------------------------------

def test_last_confirmed_wins_high_confidence():
    r = _enrich(
        last_confirmed="2026-06-16 10:00:00",
        first_seen="2026-01-01",
        created_at="2025-12-01",
    )
    assert r["temporal_confidence"] == "high"
    assert r["effective_ts"].startswith("2026-06-16")


def test_first_seen_used_when_no_last_confirmed():
    r = _enrich(first_seen="2026-06-10 00:00:00", created_at="2025-01-01")
    assert r["temporal_confidence"] == "high"
    assert r["effective_ts"].startswith("2026-06-10")


def test_created_at_only_is_low_confidence_and_zero_bonus():
    r = _enrich(created_at="2026-06-17 09:00:00")
    assert r["temporal_confidence"] == "low"
    assert r["recency_bonus"] == 0.0  # weak timestamp never manufactures lift


def test_undated_item():
    r = _enrich(text="no dates here")
    assert r["temporal_confidence"] == "none"
    assert r["temporal_bucket"] == "undated"
    assert r["effective_ts"] is None
    assert r["age_days"] is None
    assert r["recency_bonus"] == 0.0


# --- buckets ---------------------------------------------------------------

@pytest.mark.parametrize("days,expected", [
    (0.2, "today"),
    (3.0, "this_week"),
    (20.0, "this_month"),
    (120.0, "older"),
])
def test_temporal_buckets(days, expected):
    ts = (AS_OF - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    r = _enrich(last_confirmed=ts)
    assert r["temporal_bucket"] == expected


# --- recency decay ---------------------------------------------------------

def test_recency_decay_monotonic_decreasing():
    fresh = T.recency_bonus(0.0, "high")
    half = T.recency_bonus(T.DEFAULT_HALF_LIFE_DAYS, "high")
    old = T.recency_bonus(60.0, "high")
    assert fresh > half > old


def test_recency_half_life_value():
    # At exactly half-life, bonus == weight * 0.5
    half = T.recency_bonus(T.DEFAULT_HALF_LIFE_DAYS, "high")
    assert abs(half - (T.DEFAULT_RECENCY_WEIGHT * 0.5)) < 1e-9


def test_weight_zero_reproduces_legacy_ordering():
    # Regression safety: weight=0 must contribute exactly nothing.
    assert T.recency_bonus(1.0, "high", weight=0.0) == 0.0


def test_low_and_none_confidence_get_no_bonus():
    assert T.recency_bonus(1.0, "low") == 0.0
    assert T.recency_bonus(1.0, "none") == 0.0


# --- clock skew / robustness ----------------------------------------------

def test_future_timestamp_clamps_age_to_zero():
    r = _enrich(last_confirmed="2027-01-01 00:00:00")
    assert r["age_days"] == 0.0
    # Bonus capped at full weight (age 0), never inflated beyond it.
    assert r["recency_bonus"] <= T.DEFAULT_RECENCY_WEIGHT + 1e-9


def test_garbage_timestamp_treated_as_undated():
    r = _enrich(last_confirmed="not-a-date")
    assert r["temporal_confidence"] == "none"
    assert r["effective_ts"] is None


def test_naive_datetime_assumed_utc():
    r = _enrich(last_confirmed=datetime(2026, 6, 17, 21, 50))  # naive
    assert r["effective_ts"].endswith("+00:00")
    assert r["age_days"] == pytest.approx(1.0, abs=0.05)


# --- additivity ------------------------------------------------------------

def test_enrichment_is_additive_and_nondestructive():
    src = {"text": "hi", "score": 0.9, "last_confirmed": "2026-06-18 00:00:00"}
    r = T.enrich_item(src, AS_OF)
    assert r["text"] == "hi"
    assert r["score"] == 0.9
    for k in ("effective_ts", "age_days", "temporal_bucket",
              "temporal_label", "temporal_confidence", "recency_bonus"):
        assert k in r


def test_enrich_rows_processes_all():
    rows = [
        {"last_confirmed": "2026-06-18 00:00:00"},
        {"text": "undated"},
    ]
    out = T.enrich_rows(rows, AS_OF)
    assert out is rows  # in place
    assert rows[0]["temporal_bucket"] == "today"
    assert rows[1]["temporal_bucket"] == "undated"


# --- human labels ----------------------------------------------------------

@pytest.mark.parametrize("days,expected_contains", [
    (2.0, "day"),
    (10.0, "week"),
    (60.0, "month"),
    (400.0, "year"),
])
def test_human_labels(days, expected_contains):
    assert expected_contains in T.temporal_label(days)


# ===========================================================================
# Phase T2 — absolute recency-decay ranking contract
# ===========================================================================
# These tests validate the SCORING MATH that the SQL `abs_recency_bonus` term
# mirrors (weight * 0.5^(age/half_life), high-confidence only). The live SQL
# ranking against Postgres is the tester's job; here we lock the contract so
# the SQL formula has a deterministic spec to match.

def _final_score(fts, vec, struct, importance, abs_recency):
    """Mirror of the SQL ranked.final_score composition (sans relative bonus)."""
    return (fts * 3.0) + (vec * 1.1) + struct + importance + abs_recency


def test_t2_recency_breaks_ties_between_equal_relevance():
    # Two items, identical relevance; fresher one must rank higher.
    fresh = T.recency_bonus(0.5, "high")
    stale = T.recency_bonus(45.0, "high")
    s_fresh = _final_score(1.0, 0.5, 0.0, 0.2, fresh)
    s_stale = _final_score(1.0, 0.5, 0.0, 0.2, stale)
    assert s_fresh > s_stale


def test_t2_relevance_primacy_strong_old_beats_weak_fresh():
    # CRITICAL guard: a strong semantic/keyword match (old) must still beat a
    # weak match (fresh). Recency must not dominate relevance.
    strong_old = _final_score(fts=1.0, vec=0.9, struct=0.0, importance=0.2,
                              abs_recency=T.recency_bonus(60.0, "high"))
    weak_fresh = _final_score(fts=0.1, vec=0.2, struct=0.0, importance=0.2,
                              abs_recency=T.recency_bonus(0.0, "high"))
    assert strong_old > weak_fresh


def test_t2_weight_zero_is_exact_legacy_score():
    # Kill switch: weight=0 => abs_recency contributes exactly 0 => legacy score.
    legacy = _final_score(1.0, 0.5, 0.1, 0.2, 0.0)
    with_zero = _final_score(1.0, 0.5, 0.1, 0.2,
                             T.recency_bonus(3.0, "high", weight=0.0))
    assert with_zero == legacy


def test_t2_low_confidence_does_not_affect_score():
    # created_at-only item gets no recency lift in ranking.
    assert T.recency_bonus(2.0, "low") == 0.0
    s_low = _final_score(1.0, 0.5, 0.0, 0.2, T.recency_bonus(2.0, "low"))
    s_none = _final_score(1.0, 0.5, 0.0, 0.2, 0.0)
    assert s_low == s_none


def test_t2_surfaced_bonus_matches_scoring_bonus():
    # The recency_bonus surfaced via enrich must equal the value used in ranking
    # for the same weight/half_life (no drift between display and score).
    item = {"last_confirmed": (AS_OF - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")}
    enriched = T.enrich_item(dict(item), AS_OF, weight=0.6, half_life_days=14.0)
    direct = round(T.recency_bonus(7.0, "high", weight=0.6, half_life_days=14.0), 4)
    assert enriched["recency_bonus"] == direct


# ===========================================================================
# Phase T3 — temporal-window filters (since / until / as_of)
# ===========================================================================
# parse_ts is the single source of truth both push-down paths use. These tests
# lock its contract + the effective-ts window semantics used by the post-fetch
# (qdrant/graph) filter path. SQL push-down is validated against live PG by the
# tester; here we pin the parsing + comparison logic deterministically.

def test_t3_parse_ts_date_only():
    dt = T.parse_ts("2026-06-01")
    assert dt is not None and dt.year == 2026 and dt.month == 6 and dt.day == 1
    assert dt.tzinfo is not None  # always aware (UTC)


def test_t3_parse_ts_iso_with_t_and_z():
    dt = T.parse_ts("2026-06-18T22:00:00Z")
    assert dt is not None and dt.hour == 22 and dt.tzinfo is not None


def test_t3_parse_ts_garbage_and_empty_return_none():
    for bad in ("not-a-date", "", None, "2026-13-99"):
        assert T.parse_ts(bad) is None


def test_t3_window_semantics_on_effective_ts():
    # Build rows at known effective timestamps and apply the same comparison
    # the post-fetch filter uses: keep iff (eff is None) or within [since,until].
    def eff(row):
        d, _ = T.resolve_effective_ts(row)
        return d

    def passes(row, since=None, until=None, as_of=None):
        d = eff(row)
        if d is None:
            return True  # undated rows are kept
        if since and d < T.parse_ts(since):
            return False
        if until and d > T.parse_ts(until):
            return False
        if as_of and d > T.parse_ts(as_of):
            return False
        return True

    older = {"last_confirmed": "2026-06-01 00:00:00"}
    newer = {"last_confirmed": "2026-06-17 00:00:00"}
    undated = {"text": "no timestamps here"}

    # since cuts off older
    assert passes(newer, since="2026-06-10")
    assert not passes(older, since="2026-06-10")
    # until cuts off newer
    assert passes(older, until="2026-06-10")
    assert not passes(newer, until="2026-06-10")
    # as_of excludes memories newer than the point-in-time
    assert passes(older, as_of="2026-06-10")
    assert not passes(newer, as_of="2026-06-10")
    # undated always kept (non-destructive)
    assert passes(undated, since="2026-06-10", until="2026-06-12", as_of="2026-06-11")


def test_t3_effective_ts_uses_coalesce_order_for_windows():
    # last_confirmed wins over first_seen/created_at for window placement.
    row = {
        "last_confirmed": "2026-06-17 00:00:00",
        "first_seen": "2026-01-01 00:00:00",
        "created_at": "2025-01-01 00:00:00",
    }
    d, conf = T.resolve_effective_ts(row)
    assert conf == "high"
    assert d == T.parse_ts("2026-06-17")


# ===========================================================================
# Phase T4 — temporal cohort grouping (read-only view over results)
# ===========================================================================

def _row(days=None, **kw):
    r = dict(kw)
    if days is not None:
        r["last_confirmed"] = (AS_OF - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    return r


def test_t4_cohort_counts_and_order():
    rows = [_row(0.2), _row(3), _row(20), _row(200), _row(), _row(5), _row(0.5)]
    T.enrich_rows(rows, AS_OF)
    c = T.temporal_cohorts(rows, AS_OF)
    assert c["counts"] == {"today": 2, "this_week": 2, "this_month": 1, "older": 1, "undated": 1}
    # order must follow COHORT_ORDER (newest-first, undated last), non-empty only
    assert c["order"] == ["today", "this_week", "this_month", "older", "undated"]


def test_t4_empty_cohorts_omitted():
    rows = [_row(0.1), _row(0.3)]  # all today
    T.enrich_rows(rows, AS_OF)
    c = T.temporal_cohorts(rows, AS_OF)
    assert c["order"] == ["today"]
    assert set(c["counts"]) == {"today"}
    assert "older" not in c["buckets"]


def test_t4_item_indexes_map_back_to_rows():
    rows = [_row(0.2), _row(40), _row(2)]  # today, older, this_week
    T.enrich_rows(rows, AS_OF)
    c = T.temporal_cohorts(rows, AS_OF)
    assert c["buckets"]["today"]["item_indexes"] == [0]
    assert c["buckets"]["older"]["item_indexes"] == [1]
    assert c["buckets"]["this_week"]["item_indexes"] == [2]


def test_t4_newest_oldest_labels_within_cohort():
    rows = [_row(2), _row(5)]  # both this_week
    T.enrich_rows(rows, AS_OF)
    c = T.temporal_cohorts(rows, AS_OF)
    b = c["buckets"]["this_week"]
    assert b["count"] == 2
    assert b["newest_label"] == "2 days ago"
    assert b["oldest_label"] == "5 days ago"


def test_t4_does_not_reorder_or_mutate_rows():
    rows = [_row(40), _row(0.1), _row(5)]
    before = [dict(r) for r in rows]
    T.enrich_rows(rows, AS_OF)
    enriched_snapshot = [dict(r) for r in rows]
    _ = T.temporal_cohorts(rows, AS_OF)
    # cohort view must NOT change row order or contents
    assert rows == enriched_snapshot
    # original positional order preserved (older still first)
    assert rows[0]["last_confirmed"] == before[0]["last_confirmed"]


def test_t4_works_without_prior_enrichment():
    # If rows weren't enriched, temporal_cohorts computes buckets on the fly.
    rows = [_row(0.2), _row(40)]
    c = T.temporal_cohorts(rows, AS_OF)
    assert c["counts"].get("today") == 1
    assert c["counts"].get("older") == 1


def test_t4_empty_input():
    c = T.temporal_cohorts([], AS_OF)
    assert c == {"order": [], "counts": {}, "buckets": {}}


# ===========================================================================
# Phase T5 — temporal observability signals
# ===========================================================================

def test_t5_signals_basic_counts_and_stats():
    rows = [_row(0.2), _row(3), _row(20), _row(), _row(5)]
    T.enrich_rows(rows, AS_OF)
    s = T.temporal_signals(rows)
    assert s["n"] == 5
    assert s["dated"] == 4
    assert s["undated"] == 1
    assert s["age_days"]["min"] == 0.2
    assert s["age_days"]["max"] == 20.0
    assert "median" in s["age_days"]
    assert s["confidence"]["high"] == 4
    assert s["confidence"]["none"] == 1


def test_t5_signals_empty_input_safe():
    s = T.temporal_signals([])
    assert s["n"] == 0 and s["dated"] == 0 and s["undated"] == 0
    assert s["age_days"] is None and s["recency_bonus"] is None
    assert s["confidence"] == {"high": 0, "low": 0, "none": 0}


def test_t5_signals_confidence_breakdown():
    # last_confirmed => high, created_at-only => low, none => none
    rows = [
        {"last_confirmed": "2026-06-17 00:00:00"},
        {"created_at": "2026-06-10 00:00:00"},
        {"text": "no ts"},
    ]
    s = T.temporal_signals(rows)  # intentionally NOT enriched first
    assert s["confidence"] == {"high": 1, "low": 1, "none": 1}
    assert s["dated"] == 2   # high + low are dated
    assert s["undated"] == 1


def test_t5_signals_recency_bonus_only_from_enriched():
    rows = [_row(1), _row(10)]
    T.enrich_rows(rows, AS_OF)
    s = T.temporal_signals(rows)
    assert s["recency_bonus"] is not None
    assert 0.0 <= s["recency_bonus"]["min"] <= s["recency_bonus"]["max"] <= 0.6


def test_t5_signals_median_even_and_odd():
    # odd count
    rows_odd = [_row(1), _row(2), _row(3)]
    T.enrich_rows(rows_odd, AS_OF)
    assert T.temporal_signals(rows_odd)["age_days"]["median"] == 2.0
    # even count -> average of two middle
    rows_even = [_row(1), _row(2), _row(3), _row(4)]
    T.enrich_rows(rows_even, AS_OF)
    assert T.temporal_signals(rows_even)["age_days"]["median"] == 2.5
