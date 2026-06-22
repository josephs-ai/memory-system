"""
consolidate_memory_digest.py — Phase 4 "auto_dream" consolidation.

Roll the per-session / daily session cards (Phase 2) UP into durable, deduped,
topic DIGEST nodes under the canonical tree, cross-linked with [[wikilinks]], so
long-term memory COMPOUNDS instead of merely accumulating. This is the
consolidation pass a human does when reviewing their journal and distilling it.

Pipeline position: session -> daily -> DIGEST -> boot pack.

Design (consistent with Phase 2/3):
- NO LLM in the derivation path: deterministic routing + dedup + link detection.
- Reuse existing seams: consolidate_project_daily_memory.collect_bullets /
  unique_preserve_order for bullet extraction + dedup; canonical_memory_paths for
  placement. (memory_knowledge_graph link detectors operate on DB-item dicts, a
  heavier shape than our markdown bullets, so wikilinking here is a lightweight
  deterministic pass over digest nodes; Neo4j sync stays an optional follow-up.)
- Provenance + freshness on every node (constraint 3). Idempotent upsert: re-running
  folds new days in without duplicating understanding (content-keyed dedup).
- DEGRADED/loud over silent: unroutable items go to systems/misc, never dropped.

Output: canonical/digest/<kind>/<slug>.md  + canonical/metadata/catalogs/digest_index.md
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import canonical_memory_paths as cmp
from consolidate_project_daily_memory import collect_bullets, unique_preserve_order

FRESH, STALE, DEGRADED = "fresh", "stale", "degraded"

# A digest "topic" is (kind, slug). Cards already separate sections, so routing is
# mostly structural. Each entry: (digest_kind, slug, section-or-keyword matchers).
# Section names come from the Phase 2 card renderer.
SECTION_TO_KIND = {
    "decisions": ("decisions", "decisions"),
    "blockers": ("bugs", "blockers"),
    "tests run": ("systems", "tests-and-tooling"),
    "files changed": ("systems", "code-changes"),
    "summary": ("procedure", "session-summaries"),
    "next step": ("decisions", "open-threads"),
}

# Sections whose bullets are pure machine metadata, not knowledge. The session
# card's Provenance (transcript path / event count / span) and roll-up scaffolding
# are already captured structurally in each digest node's own provenance, so their
# bullets must NOT flow into the compounded understanding (they were polluting
# systems/misc with file paths + event counts). Matched case-insensitively; a
# "## Session <id>" roll-up header is also metadata.
_IGNORE_SECTIONS = {"provenance", "day index"}
_IGNORE_SECTION_PREFIXES = ("session ", "daily roll-up", "boot pack")

# Keyword fallbacks for bullets that arrive without a clear section context.
_KEYWORD_KIND = [
    (re.compile(r"\b(decide|decision|approv|will|plan to|agreed)\b", re.I), ("decisions", "decisions")),
    (re.compile(r"\b(bug|fail|error|broken|blocked|regression|exploit)\b", re.I), ("bugs", "blockers")),
    (re.compile(r"\b(test|pytest|build|ci|lint|health)\b", re.I), ("systems", "tests-and-tooling")),
    (re.compile(r"\b(prefer|like|dislike|annoy|user wants)\b", re.I), ("personal", "user-preferences")),
    (re.compile(r"\bhttps?://|\bdoc(s|umentation)\b", re.I), ("resources", "links-and-docs")),
]
DEFAULT_KIND = ("systems", "misc")


@dataclass
class DigestNode:
    kind: str
    slug: str
    bullets: list[str] = field(default_factory=list)
    source_days: set[str] = field(default_factory=set)
    source_cards: int = 0


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def route_section(section_title: str) -> tuple[str, str]:
    key = section_title.strip().lower().lstrip("# ").strip()
    return SECTION_TO_KIND.get(key, ("", ""))


def route_bullet(bullet: str) -> tuple[str, str]:
    for rx, kind_slug in _KEYWORD_KIND:
        if rx.search(bullet):
            return kind_slug
    return DEFAULT_KIND


# ---------------------------------------------------------------------------
# Card parsing → routed bullets
# ---------------------------------------------------------------------------
_SECTION_RE = re.compile(r"^#{1,3}\s+(.*)$")


def _iter_card_sections(text: str):
    """Yield (section_title, [bullets]) from a daily roll-up / session card."""
    current = None
    buf: list[str] = []
    for line in text.splitlines():
        m = _SECTION_RE.match(line)
        if m:
            if current is not None:
                yield current, buf
            current = m.group(1).strip()
            buf = []
        elif line.strip().startswith("- "):
            buf.append(line.strip())
    if current is not None:
        yield current, buf


def _is_ignored_section(title: str) -> bool:
    key = title.strip().lower().lstrip("# ").strip()
    if key in _IGNORE_SECTIONS:
        return True
    return any(key.startswith(p) for p in _IGNORE_SECTION_PREFIXES)


def route_card_text(text: str) -> dict[tuple[str, str], list[str]]:
    """Map a card/roll-up's bullets into digest (kind, slug) buckets.

    Metadata sections (Provenance, roll-up/session headers) are skipped entirely
    so machine bookkeeping (transcript paths, event counts, spans) never pollutes
    the compounded understanding — that bookkeeping lives in each node's own
    provenance, derived structurally."""
    buckets: dict[tuple[str, str], list[str]] = {}
    for title, bullets in _iter_card_sections(text):
        if _is_ignored_section(title):
            continue
        kind, slug = route_section(title)
        for b in bullets:
            target = (kind, slug) if kind else route_bullet(b)
            buckets.setdefault(target, []).append(b)
    return buckets


# ---------------------------------------------------------------------------
# Digest node read/merge/render (idempotent upsert)
# ---------------------------------------------------------------------------
_FM_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _read_existing_node(kind: str, slug: str) -> tuple[list[str], set[str]]:
    """Return (existing_bullets, existing_source_days) from a digest node so an
    upsert folds new content in without losing or duplicating prior understanding."""
    path = cmp.digest_node(kind, slug, ensure=False)
    if not path.exists():
        return [], set()
    text = path.read_text(encoding="utf-8", errors="ignore")
    days: set[str] = set()
    m = _FM_RE.match(text)
    if m:
        for line in m.group(1).splitlines():
            if line.startswith("source_days:"):
                raw = line.split(":", 1)[1].strip().strip("[]")
                days = {d.strip() for d in raw.split(",") if d.strip()}
    bullets: list[str] = []
    in_understanding = False
    for line in text.splitlines():
        if line.startswith("## Current understanding"):
            in_understanding = True
            continue
        if in_understanding:
            if line.startswith("## "):
                break
            if line.strip().startswith("- "):
                bullets.append(line.strip())
    return bullets, days


def upsert_digest_node(kind: str, slug: str, new_bullets: list[str],
                       source_days: set[str], source_cards: int) -> DigestNode:
    existing_bullets, existing_days = _read_existing_node(kind, slug)
    # content-keyed dedup: preserve prior order, append genuinely new bullets
    merged = unique_preserve_order(existing_bullets + new_bullets)
    node = DigestNode(
        kind=kind, slug=slug, bullets=merged,
        source_days=existing_days | set(source_days),
        source_cards=source_cards,
    )
    return node


# ---------------------------------------------------------------------------
# Wikilinks (lightweight, deterministic, over digest nodes)
# ---------------------------------------------------------------------------
def compute_wikilinks(node: DigestNode, all_nodes: list[tuple[str, str]]) -> list[str]:
    """Link a node to sibling nodes it likely relates to: same kind (different
    slug) + cross-kind topic-word overlap. Deterministic, no model."""
    links: list[str] = []
    node_words = set(re.findall(r"[a-z0-9]+", node.slug.lower()))
    for (k, s) in all_nodes:
        if (k, s) == (node.kind, node.slug):
            continue
        sib_words = set(re.findall(r"[a-z0-9]+", s.lower()))
        same_kind = (k == node.kind)
        overlap = bool(node_words & sib_words)
        if same_kind or overlap:
            wl = f"[[{k}/{s}]]"
            if wl not in links:
                links.append(wl)
    return links


def render_digest_node(node: DigestNode, links: list[str], *, freshness: str) -> str:
    src_through = max(node.source_days) if node.source_days else "?"
    fm = [
        "---",
        f"kind: {node.kind}",
        f"slug: {node.slug}",
        f"generated_at: {_now_iso()}",
        f"freshness: {freshness}",
        f"source_through: {src_through}",
        f"source_days: [{', '.join(sorted(node.source_days))}]",
        f"source_cards: {node.source_cards}",
        # links are wikilinks ([[kind/slug]]); list them space-separated to avoid
        # a confusing [[[...]]] collision with a YAML [..] list wrapper.
        f"links: {' '.join(links) if links else '-'}",
        "---",
        "",
        f"# {node.kind.title()}: {node.slug.replace('-', ' ')}",
        "",
        "## Current understanding",
    ]
    body = list(node.bullets) if node.bullets else ["_(no items yet)_"]
    tail = [""]
    if links:
        tail += ["## Related", *links, ""]
    tail += [
        "## Provenance",
        f"- source days: {', '.join(sorted(node.source_days)) or '-'}",
        f"- contributing cards: {node.source_cards}",
        "",
    ]
    return "\n".join(fm + body + tail).rstrip() + "\n"


def write_digest_node(node: DigestNode, links: list[str], *, freshness: str = FRESH) -> Path:
    out = cmp.digest_node(node.kind, node.slug)
    out.write_text(render_digest_node(node, links, freshness=freshness), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# auto_dream driver
# ---------------------------------------------------------------------------
def _recent_rollups(since_days: float, agent: str | None) -> list[tuple[str, Path]]:
    """(day, path) for daily agent roll-ups within the window."""
    base = cmp.canonical_root() / "daily"
    rows: list[tuple[str, Path]] = []
    if not base.is_dir():
        return rows
    cutoff = (datetime.now(timezone.utc).date()
              - __import__("datetime").timedelta(days=int(since_days)))
    for ddir in sorted(base.iterdir()):
        if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", ddir.name):
            continue
        try:
            d = datetime.strptime(ddir.name, "%Y-%m-%d").date()
        except ValueError:
            continue
        if d < cutoff:
            continue
        for card in ddir.glob("*.md"):
            if card.name == "_index.md":
                continue
            if agent and card.stem != agent:
                continue
            rows.append((ddir.name, card))
    return rows


def run_auto_dream(since_days: float = 7, agent: str | None = None,
                   dry_run: bool = False) -> dict:
    cmp.scaffold(write_readme=True)
    rollups = _recent_rollups(since_days, agent)

    # Aggregate routed bullets across all rollups, tracking source days/card counts.
    agg: dict[tuple[str, str], dict] = {}
    for day, card in rollups:
        text = card.read_text(encoding="utf-8", errors="ignore")
        for (kind, slug), bullets in route_card_text(text).items():
            entry = agg.setdefault((kind, slug), {"bullets": [], "days": set(), "cards": 0})
            entry["bullets"].extend(bullets)
            entry["days"].add(day)
            entry["cards"] += 1

    if dry_run:
        return {
            "dry_run": True,
            "rollups": len(rollups),
            "nodes": [{"kind": k, "slug": s, "bullets": len(v["bullets"]),
                       "days": sorted(v["days"])} for (k, s), v in sorted(agg.items())],
        }

    # Upsert nodes (idempotent merge with any existing node).
    nodes: list[DigestNode] = []
    for (kind, slug), v in agg.items():
        deduped = unique_preserve_order(v["bullets"])
        node = upsert_digest_node(kind, slug, deduped, v["days"], v["cards"])
        nodes.append(node)

    all_keys = [(n.kind, n.slug) for n in nodes]
    # also include pre-existing nodes on disk so links resolve across runs
    all_keys = unique_preserve_order(all_keys + _existing_node_keys())

    written = []
    for node in nodes:
        links = compute_wikilinks(node, all_keys)
        path = write_digest_node(node, links)
        written.append({"kind": node.kind, "slug": node.slug,
                        "bullets": len(node.bullets),
                        "source_days": sorted(node.source_days),
                        "links": len(links), "path": str(path)})

    index_path = write_digest_index(all_keys)
    return {"dry_run": False, "rollups": len(rollups),
            "nodes_written": len(written), "nodes": written,
            "index": str(index_path)}


def _existing_node_keys() -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    for kind in cmp.DIGEST_KINDS:
        d = cmp.digest_dir(kind, ensure=False)
        if d.is_dir():
            for f in sorted(d.glob("*.md")):
                keys.append((kind, f.stem))
    return keys


def write_digest_index(all_keys: list[tuple[str, str]]) -> Path:
    out = cmp.metadata_dir("catalogs") / "digest_index.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Digest Index", "", f"_Generated {_now_iso()} — {len(all_keys)} node(s)._", ""]
    by_kind: dict[str, list[str]] = {}
    for kind, slug in sorted(all_keys):
        by_kind.setdefault(kind, []).append(slug)
    for kind in sorted(by_kind):
        lines.append(f"## {kind}")
        for slug in by_kind[kind]:
            lines.append(f"- [[{kind}/{slug}]]")
        lines.append("")
    out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="auto_dream: consolidate cards into digest nodes")
    ap.add_argument("--since-days", type=float, default=7)
    ap.add_argument("--agent")
    ap.add_argument("--all-agents", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    agent = None if args.all_agents else args.agent
    result = run_auto_dream(since_days=args.since_days, agent=agent, dry_run=args.dry_run)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if result.get("dry_run"):
            print(f"[dry-run] {result['rollups']} rollup(s) -> {len(result['nodes'])} digest node(s):")
            for n in result["nodes"]:
                print(f"  {n['kind']}/{n['slug']}: {n['bullets']} bullets, days {n['days']}")
        else:
            print(f"{result['rollups']} rollup(s) -> {result['nodes_written']} node(s) written")
            for n in result["nodes"]:
                print(f"  {n['kind']}/{n['slug']}: {n['bullets']} bullets, "
                      f"{n['links']} links, days {n['source_days']}")
            print(f"index: {result['index']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
