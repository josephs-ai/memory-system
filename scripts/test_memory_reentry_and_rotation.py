"""
Regression tests for two memory-loss bugs:

1. Discard-ban: once-discarded claims were permanently suppressed because the
   id-dedup gate queried memory_discarded and candidate_id is a deterministic
   hash of (chunk|claim_text). Plus the content-match gate never aged out and
   ignored confidence. Fixes: drop discarded from id-dedup; window the
   content-match gate by recency; allow confidence override; soft-hold
   below-threshold items instead of hard-discarding.

2. Rotation loss: rotated "*.jsonl.deleted.*" transcripts with an uncommitted
   tail were never checkpointed (discovery globbed only "*.jsonl"). Fix:
   find_pending_transcripts_for_agent includes recent rotated files that still
   have bytes past their cursor.
"""
import importlib
import json
import os
import time
from pathlib import Path

import route_memory_items_batch as R


# --------------------------------------------------------------------------
# Fix 1: rejection re-entry semantics
# --------------------------------------------------------------------------

def _idx(disc):
    return R.build_rejected_index(disc)


def test_low_confidence_reextraction_still_suppressed():
    tm, sm = _idx([{"text": "x likes y", "confidence": 0.3}])
    assert R.fast_candidate_matches_rejected({"text": "x likes y", "confidence": 0.35}, tm, sm) is True


def test_higher_confidence_reextraction_allowed_back():
    tm, sm = _idx([{"text": "x likes y", "confidence": 0.3}])
    # +0.15 margin default → 0.9 clearly exceeds → not suppressed
    assert R.fast_candidate_matches_rejected({"text": "x likes y", "confidence": 0.9}, tm, sm) is False


def test_slot_match_suppressed_and_overridable():
    disc = [{"entity": "user", "property": "theme", "value": "dark",
             "scope": "stable", "confidence": 0.2}]
    tm, sm = _idx(disc)
    low = {"entity": "user", "property": "theme", "value": "dark",
           "scope": "stable", "confidence": 0.25}
    high = {"entity": "user", "property": "theme", "value": "dark",
            "scope": "stable", "confidence": 0.95}
    assert R.fast_candidate_matches_rejected(low, tm, sm) is True
    assert R.fast_candidate_matches_rejected(high, tm, sm) is False


def test_missing_prior_confidence_is_non_overridable():
    # A discard with no stored confidence must NOT be overridable by the
    # default extractor confidence (~0.5); it stays suppressed for the window.
    tm, sm = _idx([{"text": "x likes y"}])  # no 'confidence' key
    assert R.fast_candidate_matches_rejected({"text": "x likes y", "confidence": 0.5}, tm, sm) is True
    assert R.fast_candidate_matches_rejected({"text": "x likes y", "confidence": 0.99}, tm, sm) is True


def test_explicit_zero_confidence_is_overridable():
    # An explicit 0.0 (vs missing) is still overridable by stronger evidence.
    tm, sm = _idx([{"text": "x likes y", "confidence": 0.0}])
    assert R.fast_candidate_matches_rejected({"text": "x likes y", "confidence": 0.9}, tm, sm) is False


def test_window_disabled_never_suppresses(monkeypatch):
    monkeypatch.setattr(R, "REJECT_WINDOW_DAYS", 0)
    tm, sm = _idx([{"text": "x likes y", "confidence": 0.0}])
    assert R.fast_candidate_matches_rejected({"text": "x likes y", "confidence": 0.0}, tm, sm) is False


def test_id_dedup_excludes_discarded():
    # The id-dedup SQL must not reference memory_discarded; discards are handled
    # by the rejection gate, not the hard id skip.
    import inspect
    src = inspect.getsource(R.bulk_fetch_existing_ids)
    # Strip the docstring (which legitimately explains the exclusion) and assert
    # the executable body issues no SELECT against memory_discarded.
    body = src.split('"""')[-1]
    assert "FROM memory_discarded" not in body


def test_below_threshold_soft_holds_to_inbox(monkeypatch):
    # Force a deterministic policy and ensure a sub-inbox-threshold item is
    # routed to INBOX (soft hold), not DISCARDED, by default.
    policy = {
        "default": {
            "allowed_memory_types": ["fact"],
            "allowed_scopes": ["stable"],
            "min_confidence_auto": 0.9,
            "min_confidence_inbox": 0.6,
        }
    }
    monkeypatch.setattr(R, "load_agent_policy", lambda: policy)
    monkeypatch.delenv("OPENCLAW_HARD_DISCARD_BELOW_THRESHOLD", raising=False)
    route, reason = R.apply_agent_policy(
        {"memory_type": "fact", "scope": "stable", "confidence": 0.1}
    )
    assert route == "INBOX"
    assert "soft_hold" in reason


def test_below_threshold_hard_discard_opt_in(monkeypatch):
    policy = {
        "default": {
            "allowed_memory_types": ["fact"],
            "allowed_scopes": ["stable"],
            "min_confidence_auto": 0.9,
            "min_confidence_inbox": 0.6,
        }
    }
    monkeypatch.setattr(R, "load_agent_policy", lambda: policy)
    monkeypatch.setenv("OPENCLAW_HARD_DISCARD_BELOW_THRESHOLD", "1")
    route, _ = R.apply_agent_policy(
        {"memory_type": "fact", "scope": "stable", "confidence": 0.1}
    )
    assert route == "DISCARDED"


# --------------------------------------------------------------------------
# Fix 2: rotated transcript discovery
# --------------------------------------------------------------------------

import checkpoint_agent as C


def _make_agent(tmp_path, name="agentx"):
    sd = tmp_path / "agents" / name / "sessions"
    sd.mkdir(parents=True)
    return sd


def test_rotated_with_uncommitted_tail_is_pending(tmp_path, monkeypatch):
    sd = _make_agent(tmp_path)
    monkeypatch.setattr(C, "AGENTS_DIR", tmp_path / "agents")
    # No cursor → uncommitted by definition.
    rotated = sd / "abc.jsonl.deleted.2026-06-20T00-00-00.000Z"
    rotated.write_text("line1\nline2\n", encoding="utf-8")
    pend = C.find_pending_transcripts_for_agent("agentx")
    assert rotated in pend


def test_rotated_fully_committed_is_not_pending(tmp_path, monkeypatch):
    sd = _make_agent(tmp_path, "agenty")
    monkeypatch.setattr(C, "AGENTS_DIR", tmp_path / "agents")
    rotated = sd / "def.jsonl.deleted.2026-06-20T00-00-00.000Z"
    data = "line1\nline2\n"
    rotated.write_text(data, encoding="utf-8")
    # Cursor committed to full size → already ingested → not pending.
    monkeypatch.setattr(C, "load_cursor",
                        lambda agent, t=None: {"committed_offset": len(data.encode())})
    pend = C.find_pending_transcripts_for_agent("agenty")
    assert rotated not in pend


def test_rotated_outside_lookback_skipped(tmp_path, monkeypatch):
    sd = _make_agent(tmp_path, "agentz")
    monkeypatch.setattr(C, "AGENTS_DIR", tmp_path / "agents")
    monkeypatch.setattr(C, "ROTATED_LOOKBACK_DAYS", 1)
    rotated = sd / "old.jsonl.deleted.2026-01-01T00-00-00.000Z"
    rotated.write_text("line1\nline2\n", encoding="utf-8")
    old = time.time() - 5 * 86400
    os.utime(rotated, (old, old))
    monkeypatch.setattr(C, "load_cursor", lambda agent, t=None: None)
    pend = C.find_pending_transcripts_for_agent("agentz")
    assert rotated not in pend


def test_active_session_always_included(tmp_path, monkeypatch):
    sd = _make_agent(tmp_path, "agenta")
    monkeypatch.setattr(C, "AGENTS_DIR", tmp_path / "agents")
    active = sd / "live.jsonl"
    active.write_text("line1\n", encoding="utf-8")
    pend = C.find_pending_transcripts_for_agent("agenta")
    assert active in pend
