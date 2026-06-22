#!/usr/bin/env python3
"""
Memory system utility: build a searchable index of the READABLE memory layer.

Phase 5 (card search). Walks the canonical readable tree —
  - session/<agent>/<session_id>.md   (Phase 2 session cards)
  - daily/<day>/<agent>.md            (Phase 2 daily roll-ups)
  - digest/<kind>/<slug>.md           (Phase 4 digest nodes)
and emits a compact, deterministic JSONL index at
  metadata/indexes/card_index.jsonl

The index is the substrate for search_memory_cards.py. It is intentionally
dependency-light (stdlib only) and frontmatter-driven so that the readable layer
stays searchable EVEN WHEN the embedding / vector search service is down — the
exact failure class this memory upgrade hardens against.

Design rules (consistent with Phase 2-4):
  - deterministic, no LLM, no network.
  - provenance + freshness carried per record (truthful, never invented).
  - body text is bounded so the index stays small and fast to load.
  - malformed cards are skipped + counted, never fatal (loud-ish via --json).
  - idempotent: rebuilding from the same tree yields the same records
    (sorted by id), only the index's own generated_at header changes.

Key functions: build_card_index, load_index, iter_card_files, parse_card.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import canonical_memory_paths as cmp  # noqa: E402

INDEX_NAME = "card_index.jsonl"
# Bound per-record body text so the index file stays small and quick to scan.
_MAX_BODY_CHARS = 4000

_FM_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
_BULLET_RE = re.compile(r"^\s*-\s+(.*)$")
_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
_SECTION_RE = re.compile(r"^#{1,3}\s+(.*)$")


# ---------------------------------------------------------------------------
# Record model
# ---------------------------------------------------------------------------
@dataclass
class CardRecord:
    id: str
    kind: str  # session | daily | digest
    path: str  # absolute path on disk
    title: str
    text: str  # bounded body (bullets/section text), search target
    freshness: str = "unknown"
    generated_at: str = ""
    agent: str | None = None
    day: str | None = None
    digest_kind: str | None = None
    slug: str | None = None
    source_through: str | None = None
    wikilinks: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Return (frontmatter_dict, body). Frontmatter values are raw strings."""
    m = _FM_RE.match(text)
    if not m:
        return {}, text
    fm: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            fm[k.strip()] = v.strip()
    return fm, text[m.end():]


def _first_heading(body: str) -> str:
    for line in body.splitlines():
        s = line.strip()
        if s.startswith("#"):
            return s.lstrip("# ").strip()
    return ""


def _collect_searchable_text(body: str) -> str:
    """Bullets + section headings = the meaningful, low-noise search target.

    We deliberately keep bullets and (non-metadata) section titles and drop blank
    lines / prose scaffolding, then bound the result so the index stays compact.
    """
    parts: list[str] = []
    for line in body.splitlines():
        hm = _SECTION_RE.match(line)
        if hm:
            title = hm.group(1).strip()
            # skip pure-metadata scaffolding headings
            low = title.lower()
            if low.startswith(("provenance", "session ", "daily roll-up", "boot pack")):
                continue
            parts.append(title)
            continue
        bm = _BULLET_RE.match(line)
        if bm:
            parts.append(bm.group(1).strip())
    joined = "\n".join(p for p in parts if p)
    if len(joined) > _MAX_BODY_CHARS:
        joined = joined[:_MAX_BODY_CHARS]
    return joined


def _wikilinks(text: str) -> list[str]:
    seen: list[str] = []
    for m in _WIKILINK_RE.finditer(text):
        v = m.group(1).strip()
        if v and v not in seen:
            seen.append(v)
    return seen


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
def iter_card_files(root: Path | None = None):
    """Yield (kind, path) for every readable card under the canonical tree.

    kind ∈ {session, daily, digest}. Boot packs are NOT indexed (they are derived
    orientation artifacts, not source knowledge)."""
    base = root or cmp.canonical_root()
    session_root = base / "session"
    daily_root = base / "daily"
    digest_root = base / "digest"

    # Boot packs are derived orientation artifacts, never source knowledge — exclude
    # them wherever they appear, not just under session/.
    def _is_boot_pack(p: Path) -> bool:
        return p.name == "_BOOT_PACK.md"

    if session_root.is_dir():
        for p in sorted(session_root.rglob("*.md")):
            if _is_boot_pack(p):
                continue
            yield "session", p
    if daily_root.is_dir():
        for p in sorted(daily_root.rglob("*.md")):
            if _is_boot_pack(p):
                continue
            yield "daily", p
    if digest_root.is_dir():
        for p in sorted(digest_root.rglob("*.md")):
            if _is_boot_pack(p):
                continue
            yield "digest", p


def parse_card(kind: str, path: Path, root: Path) -> CardRecord | None:
    """Parse one card file into a CardRecord. Returns None on unreadable/empty."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    if not text.strip():
        return None

    fm, body = _parse_frontmatter(text)
    title = _first_heading(body) or path.stem
    searchable = _collect_searchable_text(body)
    # wikilinks may live in frontmatter (links:) and/or body
    links = _wikilinks(fm.get("links", "")) + [
        x for x in _wikilinks(body) if x not in _wikilinks(fm.get("links", ""))
    ]

    # stable, path-derived id relative to canonical root
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = Path(path.name)
    rec_id = f"{kind}:{rel.as_posix()}"

    rec = CardRecord(
        id=rec_id,
        kind=kind,
        path=str(path),
        title=title,
        text=searchable,
        freshness=fm.get("freshness", "unknown"),
        generated_at=fm.get("generated_at", ""),
        wikilinks=links,
    )

    if kind == "session":
        rec.agent = fm.get("agent") or (path.parent.name if path.parent.name != "session" else None)
        rec.day = fm.get("day")
    elif kind == "daily":
        rec.day = path.parent.name
        rec.agent = path.stem
    elif kind == "digest":
        rec.digest_kind = fm.get("kind") or path.parent.name
        rec.slug = fm.get("slug") or path.stem
        rec.source_through = fm.get("source_through")
        # daily roll-ups & session cards have no source_through; digests do
    return rec


# ---------------------------------------------------------------------------
# Build / load
# ---------------------------------------------------------------------------
def build_card_index(root: Path | None = None, *, dry_run: bool = False) -> dict:
    """(Re)build the card index. Returns a stats dict.

    Idempotent: records are sorted by id, so identical trees produce identical
    record blocks; only the header generated_at differs run-to-run.
    """
    base = root or cmp.canonical_root()
    records: list[CardRecord] = []
    skipped = 0
    for kind, path in iter_card_files(base):
        rec = parse_card(kind, path, base)
        if rec is None:
            skipped += 1
            continue
        records.append(rec)

    records.sort(key=lambda r: r.id)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    by_kind: dict[str, int] = {}
    for r in records:
        by_kind[r.kind] = by_kind.get(r.kind, 0) + 1

    stats = {
        "records": len(records),
        "skipped": skipped,
        "by_kind": by_kind,
        "generated_at": now,
    }

    if dry_run:
        stats["dry_run"] = True
        return stats

    index_dir = cmp.metadata_dir("indexes")
    index_path = index_dir / INDEX_NAME
    lines = [json.dumps({"_index_meta": stats}, ensure_ascii=False, sort_keys=True)]
    lines.extend(r.to_json() for r in records)
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    stats["path"] = str(index_path)
    return stats


def index_path() -> Path:
    return cmp.metadata_dir("indexes", ensure=False) / INDEX_NAME


def load_index() -> tuple[dict, list[CardRecord]]:
    """Load (meta, records). Returns ({}, []) if the index is absent.

    Malformed lines are skipped (never fatal) so a partially corrupt index still
    yields usable results — degraded over silent.
    """
    p = index_path()
    if not p.is_file():
        return {}, []
    meta: dict = {}
    records: list[CardRecord] = []
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        # A line can be valid JSON yet not an object (e.g. a bare scalar/list left
        # by a truncated/partial write). Only dict lines are meaningful; anything
        # else is skipped, never fatal — the whole point is the readable-search
        # path must survive a corrupt index.
        if not isinstance(obj, dict):
            continue
        if "_index_meta" in obj:
            meta = obj["_index_meta"] if isinstance(obj["_index_meta"], dict) else {}
            continue
        try:
            records.append(CardRecord(**obj))
        except (TypeError, ValueError):
            # unknown/missing/mistyped fields -> skip this record, keep going
            continue
    return meta, records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the readable-card search index.")
    ap.add_argument("--dry-run", action="store_true", help="compute stats, write nothing")
    ap.add_argument("--json", action="store_true", help="emit stats as JSON")
    args = ap.parse_args(argv)

    stats = build_card_index(dry_run=args.dry_run)
    if args.json:
        print(json.dumps(stats, ensure_ascii=False, sort_keys=True))
    else:
        kinds = ", ".join(f"{k}={v}" for k, v in sorted(stats["by_kind"].items()))
        where = stats.get("path", "(dry-run, not written)")
        print(f"card index: {stats['records']} record(s) [{kinds}], "
              f"{stats['skipped']} skipped -> {where}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
