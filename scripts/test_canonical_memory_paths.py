"""
Hermetic tests for canonical_memory_paths.py + scaffold_canonical_memory.py.

Points OPENCLAW_CANONICAL_MEMORY_ROOT at a tmp dir so no real workspace state is
touched. Proves the layout contract, idempotency, validation, and purity of
ensure=False helpers.
"""
from __future__ import annotations

import importlib

import pytest


@pytest.fixture()
def cmp_mod(tmp_path, monkeypatch):
    root = tmp_path / "canonical"
    monkeypatch.setenv("OPENCLAW_CANONICAL_MEMORY_ROOT", str(root))
    import canonical_memory_paths as cmp

    importlib.reload(cmp)  # re-read env at module level if needed
    return cmp, root


def test_root_resolves_from_env(cmp_mod):
    cmp, root = cmp_mod
    assert cmp.canonical_root() == root.resolve()


def test_scaffold_creates_full_tree(cmp_mod):
    cmp, root = cmp_mod
    result = cmp.scaffold()
    assert result["ok"]
    assert (root / "session").is_dir()
    assert (root / "daily").is_dir()
    assert (root / "resources").is_dir()
    for kind in cmp.DIGEST_KINDS:
        assert (root / "digest" / kind).is_dir()
    for name in cmp.METADATA_KINDS:
        assert (root / "metadata" / name).is_dir()
    assert (root / "README.md").exists()
    assert cmp.layout_present()


def test_scaffold_idempotent(cmp_mod):
    cmp, root = cmp_mod
    cmp.scaffold()
    second = cmp.scaffold()
    # Second run creates nothing new (README + dirs already exist)
    assert second["created"] == []
    assert cmp.layout_present()


def test_layout_present_false_before_scaffold(cmp_mod):
    cmp, root = cmp_mod
    assert not cmp.layout_present()


def test_path_helper_shapes(cmp_mod):
    cmp, root = cmp_mod
    assert cmp.session_card("Builder", "ABC-123", ensure=False) == \
        root.resolve() / "session" / "builder" / "abc-123.md"
    assert cmp.daily_agent_card("memory-coder", "2026-06-22", ensure=False) == \
        root.resolve() / "daily" / "2026-06-22" / "memory-coder.md"
    assert cmp.daily_index("2026-06-22", ensure=False) == \
        root.resolve() / "daily" / "2026-06-22" / "_index.md"
    assert cmp.digest_node("bugs", "Rotated Transcripts Not Checkpointed!", ensure=False) == \
        root.resolve() / "digest" / "bugs" / "rotated-transcripts-not-checkpointed.md"


def test_ensure_false_is_pure(cmp_mod):
    cmp, root = cmp_mod
    p = cmp.daily_dir("2026-01-01", ensure=False)
    assert not p.exists()  # must NOT have created anything


def test_invalid_digest_kind_raises(cmp_mod):
    cmp, _ = cmp_mod
    with pytest.raises(ValueError):
        cmp.digest_node("nonsense", "x", ensure=False)


def test_invalid_metadata_kind_raises(cmp_mod):
    cmp, _ = cmp_mod
    with pytest.raises(ValueError):
        cmp.metadata_dir("nope", ensure=False)


def test_invalid_day_raises(cmp_mod):
    cmp, _ = cmp_mod
    with pytest.raises(ValueError):
        cmp.daily_dir("2026/06/22", ensure=False)


def test_unicode_fullwidth_day_rejected(cmp_mod):
    # \d would have accepted fullwidth digits, creating a non-ASCII dir name.
    cmp, _ = cmp_mod
    with pytest.raises(ValueError):
        cmp.daily_dir("２０２６-０６-２２", ensure=False)


def test_empty_agent_raises(cmp_mod):
    cmp, _ = cmp_mod
    with pytest.raises(ValueError):
        cmp.session_dir("   ", ensure=False)


def test_slugify_collapses(cmp_mod):
    cmp, _ = cmp_mod
    assert cmp.slugify("  Agent  Startup // Memory Contract!! ") == "agent-startup-memory-contract"
