import os

from orchestrator.provider_config import (
    get_builder_target,
    get_reviewer_target,
    get_tester_target,
    get_writeback_target,
)


def test_builder_target_falls_back_without_tier_override(monkeypatch):
    monkeypatch.delenv("OPENCLAW_BUILDER_TARGET__CODING_STRONG", raising=False)
    monkeypatch.setenv("OPENCLAW_BUILDER_TARGET", "default.builder")
    assert get_builder_target("coding_strong") == "default.builder"


def test_builder_target_uses_tier_override(monkeypatch):
    monkeypatch.setenv("OPENCLAW_BUILDER_TARGET", "default.builder")
    monkeypatch.setenv("OPENCLAW_BUILDER_TARGET__CODING_STRONG", "tier.builder")
    assert get_builder_target("coding_strong") == "tier.builder"


def test_reviewer_target_uses_tier_override(monkeypatch):
    monkeypatch.setenv("OPENCLAW_REVIEWER_TARGET", "default.reviewer")
    monkeypatch.setenv("OPENCLAW_REVIEWER_TARGET__REVIEW_STRONG", "tier.reviewer")
    assert get_reviewer_target("review_strong") == "tier.reviewer"


def test_tester_target_uses_tier_override(monkeypatch):
    monkeypatch.setenv("OPENCLAW_TESTER_TARGET", "default.tester")
    monkeypatch.setenv("OPENCLAW_TESTER_TARGET__VALIDATION_STRONG", "tier.tester")
    assert get_tester_target("validation_strong") == "tier.tester"


def test_writeback_target_uses_tier_override(monkeypatch):
    monkeypatch.setenv("OPENCLAW_WRITEBACK_TARGET", "default.writeback")
    monkeypatch.setenv("OPENCLAW_WRITEBACK_TARGET__WRITEBACK_CHEAP", "tier.writeback")
    assert get_writeback_target("writeback_cheap") == "tier.writeback"
