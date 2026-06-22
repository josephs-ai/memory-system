"""
canonical_memory_paths.py — single source of truth for the ReMe-style canonical
memory layout (session -> daily -> digest -> resources -> metadata).

This is the readable, agent-facing layer that sits ON TOP of the backend engine
(Postgres/Qdrant/Neo4j/timeline). It is GENERATED, never hand-maintained: every
later phase imports these path helpers instead of hardcoding paths.

Layout (rooted at memory/canonical/ to avoid colliding with the legacy
memory/daily and memory/projects trees):

    memory/canonical/
      README.md                         # contract + "generated, do not edit" warning
      session/<agent>/                  # raw session pointers + per-session cards (P2)
      daily/<YYYY-MM-DD>/<agent>.md      # per-agent daily cards (P2)
      daily/<YYYY-MM-DD>/_index.md       # day roll-up (P2)
      digest/<kind>/<slug>.md            # durable digest nodes (P4); kind in DIGEST_KINDS
      resources/<YYYY-MM-DD>/            # ingested external resources (later)
      metadata/<name>/                   # generated indexes/graph/catalogs (P4/P5)

Mirrors the style of project_memory_paths.py. Root is overridable via
OPENCLAW_CANONICAL_MEMORY_ROOT for hermetic testing.
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Root resolution (env-overridable, mirrors config.py style)
# ---------------------------------------------------------------------------
_WORKSPACE = Path(
    os.environ.get("OPENCLAW_WORKSPACE", str(Path.home() / ".openclaw" / "workspace"))
)


def canonical_root() -> Path:
    """Resolve the canonical memory root at CALL time so tests/daemons that set
    the env after import still pick up the correct value."""
    return Path(
        os.environ.get(
            "OPENCLAW_CANONICAL_MEMORY_ROOT",
            str(_WORKSPACE / "memory" / "canonical"),
        )
    ).resolve()


DIGEST_KINDS: tuple[str, ...] = (
    "personal",
    "procedure",
    "projects",
    "systems",
    "bugs",
    "decisions",
    "resources",
)

METADATA_KINDS: tuple[str, ...] = ("indexes", "graph", "catalogs")

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _valid_day(day: str | None) -> str:
    day = day or today_utc()
    # Use an explicit ASCII class (not \d, which also matches Unicode/fullwidth
    # digits and would yield non-ASCII directory names once Phase 2 ingests day
    # values derived from transcript timestamps).
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", day):
        raise ValueError(f"day must be ASCII YYYY-MM-DD, got {day!r}")
    return day


def slugify(text: str) -> str:
    s = _SLUG_RE.sub("-", (text or "").strip().lower()).strip("-")
    if not s:
        raise ValueError("slug cannot be empty")
    return s


def sanitize_agent(agent: str) -> str:
    s = _SLUG_RE.sub("-", (agent or "").strip().lower()).strip("-")
    if not s:
        raise ValueError("agent cannot be empty")
    return s


def _ensure(path: Path, ensure: bool) -> Path:
    if ensure:
        path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Path helpers (ensure=True creates parents; ensure=False is pure, for tests)
# ---------------------------------------------------------------------------
def session_dir(agent: str, *, ensure: bool = True) -> Path:
    return _ensure(canonical_root() / "session" / sanitize_agent(agent), ensure)


def session_card(agent: str, session_id: str, *, ensure: bool = True) -> Path:
    sid = slugify(session_id)
    return session_dir(agent, ensure=ensure) / f"{sid}.md"


def daily_dir(day: str | None = None, *, ensure: bool = True) -> Path:
    return _ensure(canonical_root() / "daily" / _valid_day(day), ensure)


def daily_agent_card(agent: str, day: str | None = None, *, ensure: bool = True) -> Path:
    return daily_dir(day, ensure=ensure) / f"{sanitize_agent(agent)}.md"


def daily_index(day: str | None = None, *, ensure: bool = True) -> Path:
    return daily_dir(day, ensure=ensure) / "_index.md"


def digest_dir(kind: str, *, ensure: bool = True) -> Path:
    if kind not in DIGEST_KINDS:
        raise ValueError(f"unknown digest kind {kind!r}; valid: {DIGEST_KINDS}")
    return _ensure(canonical_root() / "digest" / kind, ensure)


def digest_node(kind: str, slug: str, *, ensure: bool = True) -> Path:
    return digest_dir(kind, ensure=ensure) / f"{slugify(slug)}.md"


def resources_dir(day: str | None = None, *, ensure: bool = True) -> Path:
    return _ensure(canonical_root() / "resources" / _valid_day(day), ensure)


def metadata_dir(name: str, *, ensure: bool = True) -> Path:
    if name not in METADATA_KINDS:
        raise ValueError(f"unknown metadata kind {name!r}; valid: {METADATA_KINDS}")
    return _ensure(canonical_root() / "metadata" / name, ensure)


# ---------------------------------------------------------------------------
# Scaffold
# ---------------------------------------------------------------------------
README_CONTENT = """\
# Canonical Memory Layer (GENERATED — do not hand-edit)

This tree is the readable, agent-facing memory shape that sits on top of the
backend engine (Postgres / Qdrant / Neo4j / timeline). It is **generated** by the
memory pipeline. Editing files here by hand will be overwritten and is not a
durable place to store anything.

## Shape

    session/<agent>/<session>.md      per-session cards + raw transcript pointers
    daily/<YYYY-MM-DD>/<agent>.md     per-agent daily session cards
    daily/<YYYY-MM-DD>/_index.md      per-day roll-up across agents
    digest/<kind>/<slug>.md           durable consolidated knowledge nodes
                                      kind in: personal, procedure, projects,
                                      systems, bugs, decisions, resources
    resources/<YYYY-MM-DD>/           ingested external resources (PDFs, repos, ...)
    metadata/{indexes,graph,catalogs} generated metadata

## Provenance

Every generated file carries a frontmatter/header block with `generated_at`,
the source it was derived from, and a `freshness` marker. If generation is
degraded, the file says so rather than silently presenting stale content.

## Why a separate `canonical/` root

The legacy `memory/daily/` and `memory/projects/` trees predate this layer and
are left untouched. This `canonical/` root is cleanly separable, observable by
the memory health check, and safe to regenerate.

Populated by upgrade Phases 2 (daily cards), 4 (digest/auto-dream), and later.
"""


def scaffold(*, write_readme: bool = True) -> dict:
    """Create the full empty canonical tree idempotently. Returns a summary dict."""
    root = canonical_root()
    created: list[str] = []

    def mk(p: Path) -> None:
        existed = p.exists()
        p.mkdir(parents=True, exist_ok=True)
        if not existed:
            created.append(str(p.relative_to(root)) if p != root else ".")

    mk(root)
    mk(root / "session")
    mk(root / "daily")
    for kind in DIGEST_KINDS:
        mk(root / "digest" / kind)
    mk(root / "resources")
    for name in METADATA_KINDS:
        mk(root / "metadata" / name)

    # .gitkeep so empty dirs are preservable if ever tracked
    for d in [root / "session", root / "daily", root / "resources"]:
        gk = d / ".gitkeep"
        if not gk.exists():
            gk.write_text("", encoding="utf-8")

    readme = root / "README.md"
    if write_readme and not readme.exists():
        readme.write_text(README_CONTENT, encoding="utf-8")
        created.append("README.md")

    return {"root": str(root), "created": created, "ok": True}


def layout_present() -> bool:
    """True if the canonical tree appears scaffolded (root + key subdirs exist)."""
    root = canonical_root()
    required = [
        root,
        root / "session",
        root / "daily",
        root / "digest",
        root / "resources",
        root / "metadata",
    ]
    return all(p.is_dir() for p in required)
