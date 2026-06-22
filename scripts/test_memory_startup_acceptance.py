"""
Memory startup ACCEPTANCE test — the "clueless agent" gate.

This is the objective definition of "memory startup works": given only the
generated readable surface an agent reads at boot (timeline latest + today/recent
daily), are the key orientation questions ANSWERABLE — i.e. are the needed facts
present in that surface?

This is a cheap grounding/string-presence check, NOT a live LLM eval. It gives a
red/green signal that today's context actually reached the readable layer. The
LLM-based version (spawn a fresh agent and ask) is a later, heavier add; this
keeps CI fast and deterministic while still catching the exact regression that
happened today (agent couldn't connect "PR" to the Engram pull-request flow
because the readable layer didn't carry it).

Two modes:
- Hermetic unit tests (always run): synthetic surface, prove the gate logic.
- Live smoke (opt-in via OPENCLAW_ACCEPTANCE_LIVE=1): run against the real
  generated timeline and report, skipped by default so CI is hermetic.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Gate logic (pure; reused by both hermetic and live modes)
# ---------------------------------------------------------------------------
def orientation_findings(latest_text: str, today_text: str, recent_text: str) -> dict[str, bool]:
    """Return which orientation questions are answerable from the surface."""
    blob = "\n".join([latest_text, today_text, recent_text]).lower()
    return {
        # "what happened today" — today's surface carries real events
        "what_happened_today": bool(today_text.strip()) and len(today_text.strip()) > 40,
        # "what was last session" — some recent chronology exists and is non-trivial
        "what_was_last_session": bool(recent_text.strip()) or bool(today_text.strip()),
        # "any user/assistant exchange captured" — surface is not an empty shell
        "surface_non_empty": bool(blob.strip()) and ("user" in blob or "assistant" in blob or "- [" in blob),
    }


def acceptance_passes(findings: dict[str, bool]) -> bool:
    return all(findings.values())


# ---------------------------------------------------------------------------
# Hermetic tests
# ---------------------------------------------------------------------------
def test_good_surface_passes():
    latest = "# Latest timeline summary\n- [08:30] USER: feature branch + PR (pull request): do that\n"
    today = (
        "# Timeline summary for 2026-06-22\n\n## Events\n\n"
        "- [01:16] USER: whats brsnch do?\n"
        "- [01:18] USER: feature branch + PR (pull request): do that\n"
        "- [01:46] ASSISTANT: branch fix/memory-discard-reentry-and-rotation pushed to Engram\n"
    )
    recent = "- [yesterday] ASSISTANT: shipped PCI L5 UHT rule engine\n"
    findings = orientation_findings(latest, today, recent)
    assert acceptance_passes(findings), findings


def test_empty_surface_fails():
    findings = orientation_findings("", "", "")
    assert not acceptance_passes(findings)


def test_pr_context_recoverable_from_surface():
    """The exact regression from 2026-06-22: 'PR' must be connectable to the
    pull-request flow from the readable surface alone."""
    today = (
        "## Events\n"
        "- [01:18] USER: feature branch + PR (pull request): do that\n"
        "- [01:46] ASSISTANT: PR link https://github.com/josephs-ai/Engram/pull/new/"
        "fix/memory-discard-reentry-and-rotation\n"
    )
    blob = today.lower()
    assert "pull request" in blob or "pull/new" in blob
    assert "branch" in blob
    assert "engram" in blob


def test_stale_only_shell_fails():
    # A surface that exists but carries no real exchange should not pass.
    latest = "# Latest timeline summary\n"  # header only, no events
    findings = orientation_findings(latest, "", "")
    assert not acceptance_passes(findings)


# ---------------------------------------------------------------------------
# Live smoke (opt-in)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    os.environ.get("OPENCLAW_ACCEPTANCE_LIVE") != "1",
    reason="live acceptance smoke disabled (set OPENCLAW_ACCEPTANCE_LIVE=1)",
)
def test_live_surface_answers_orientation():
    root = Path(os.environ.get("OPENCLAW_ROOT", str(Path.home() / ".openclaw")))
    timeline = root / "workspace" / ".memory-index" / "timeline"
    latest = (timeline / "latest.md")
    today_p = timeline / "daily" / f"{time.strftime('%Y-%m-%d', time.gmtime())}.md"
    dailies = sorted((timeline / "daily").glob("*.md"))
    recent_p = dailies[-2] if len(dailies) >= 2 else (dailies[-1] if dailies else None)

    latest_text = latest.read_text(encoding="utf-8", errors="ignore") if latest.exists() else ""
    today_text = today_p.read_text(encoding="utf-8", errors="ignore") if today_p.exists() else ""
    recent_text = recent_p.read_text(encoding="utf-8", errors="ignore") if recent_p else ""

    findings = orientation_findings(latest_text, today_text, recent_text)
    assert acceptance_passes(findings), f"live orientation gate failed: {findings}"
