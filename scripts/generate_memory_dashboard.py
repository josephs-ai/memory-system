#!/usr/bin/env python3
"""
Memory system utility: generate a human-readable Memory Control Panel / dashboard.

Phase 6 (readout layer). Renders the entire memory upgrade (Phases 0-5) into ONE
Markdown dashboard (+ a machine JSON twin) answering, at a glance:
  - Is my agent's memory HEALTHY? (overall + per-check, problems surfaced at top)
  - What does it currently KNOW? (agents, session cards, digests-by-topic)
  - Is the PIPELINE fresh? (session -> daily -> digest -> boot pack -> card index)

This is a READOUT only: it generates no new memory and invents nothing — every
number is gathered truthfully from on-disk artifacts. Mirrors the deterministic
"generator -> markdown" pattern of render_timeline_views.py; stdlib only; no live
server (files are the durable substrate, consistent with Phases 0-5). The emitted
JSON is the contract a future live control-panel UI can consume without reparsing.

Honesty rules:
  - Missing/stale artifacts are stated explicitly and ranked to the TOP — never a
    falsely-green panel.
  - If the in-process health run fails, fall back to the cached health JSON and
    FLAG it as cached/possibly-stale (never crash, never pretend).

Key functions: gather_inventory, build_dashboard, render_markdown, render_json.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import canonical_memory_paths as cmp  # noqa: E402

DASHBOARD_MD = "memory_dashboard.md"
DASHBOARD_JSON = "memory_dashboard.json"

_GLYPH = {"ok": "🟢", "warn": "🟡", "fail": "🔴", "skip": "⚪"}
_SEVERITY = {"ok": 0, "skip": 0, "warn": 1, "fail": 2}
_STALE_SECONDS = 24 * 3600  # pipeline stage considered stale after 24h


# ---------------------------------------------------------------------------
# Health (reuse memory_health_check; degrade to cached on failure)
# ---------------------------------------------------------------------------
def load_health(*, cached: bool) -> tuple[dict, list[str]]:
    """Return (health_dict, notes). health_dict has generated_at/overall/checks.

    cached=False -> run checks in-process (never stale). On any failure, fall back
    to the last written health JSON and flag it. cached=True -> read the JSON
    directly.
    """
    notes: list[str] = []
    import memory_health_check as mh

    if not cached:
        try:
            cfg = mh.build_cfg(None, probes={})
            report = mh.run_checks(cfg)
            return report.to_dict(), notes
        except Exception as e:  # noqa: BLE001
            notes.append(f"in-process health run failed ({e}); using cached report")

    # cached path (explicit or fallback)
    out = getattr(mh, "HEALTH_OUT", None)
    if out and Path(out).is_file():
        try:
            data = json.loads(Path(out).read_text(encoding="utf-8"))
            notes.append(f"health is CACHED (from {data.get('generated_at', '?')}) — may be stale")
            return data, notes
        except Exception as e:  # noqa: BLE001
            notes.append(f"cached health unreadable ({e})")
    notes.append("no health report available (run memory_health_check.py)")
    return {"generated_at": "", "overall": "unknown", "checks": []}, notes


# ---------------------------------------------------------------------------
# Inventory (all truthful, gathered from disk)
# ---------------------------------------------------------------------------
def _read_frontmatter_value(path: Path, key: str) -> str | None:
    try:
        head = path.read_text(encoding="utf-8", errors="ignore")[:600]
    except OSError:
        return None
    m = re.search(rf"^{re.escape(key)}:\s*(.+)$", head, re.MULTILINE)
    return m.group(1).strip() if m else None


def _age_seconds(stamp: str | None) -> float | None:
    if not stamp:
        return None
    m = re.search(r"([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]+)", stamp)
    if not m:
        # maybe a bare day
        d = re.search(r"([0-9]{4}-[0-9]{2}-[0-9]{2})", stamp)
        if not d:
            return None
        stamp = d.group(1) + "T00:00:00"
    else:
        stamp = m.group(1)
    try:
        g = datetime.fromisoformat(stamp).replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - g).total_seconds()
    except ValueError:
        return None


@dataclass
class StageState:
    name: str
    present: bool
    count: int = 0
    newest: str = ""
    stale: bool = False
    detail: str = ""

    def status(self) -> str:
        if not self.present or self.count == 0:
            return "missing"
        return "stale" if self.stale else "fresh"


@dataclass
class Inventory:
    layout_present: bool
    agents: list[dict] = field(default_factory=list)
    stages: list[StageState] = field(default_factory=list)
    digests: list[dict] = field(default_factory=list)


def _stage_from_files(name: str, files: list[Path]) -> StageState:
    if not files:
        return StageState(name=name, present=False)
    newest = ""
    for f in files:
        gen = _read_frontmatter_value(f, "generated_at") or ""
        if not gen:
            # daily roll-ups carry "_Generated <ts>_" in body, not frontmatter
            try:
                m = re.search(r"_Generated\s+([0-9T:\-Z\.]+)", f.read_text(encoding="utf-8", errors="ignore")[:200])
                gen = m.group(1) if m else ""
            except OSError:
                gen = ""
        if gen > newest:
            newest = gen
    age = _age_seconds(newest)
    return StageState(name=name, present=True, count=len(files), newest=newest,
                      stale=(age is not None and age > _STALE_SECONDS))


def gather_inventory() -> Inventory:
    if not cmp.layout_present():
        return Inventory(layout_present=False)

    root = cmp.canonical_root()
    session_root = root / "session"
    daily_root = root / "daily"
    digest_root = root / "digest"

    # per-agent session cards + boot pack
    agents: list[dict] = []
    if session_root.is_dir():
        for adir in sorted(p for p in session_root.iterdir() if p.is_dir()):
            cards = sorted(c for c in adir.glob("*.md") if c.name != "_BOOT_PACK.md")
            boot = adir / "_BOOT_PACK.md"
            # Skip empty stray dirs (no cards AND no boot pack) — not a real agent;
            # surfacing them as "agent with missing boot pack" would be misleading.
            if not cards and not boot.is_file():
                continue
            boot_gen = _read_frontmatter_value(boot, "generated_at") if boot.is_file() else None
            boot_stale = False
            if boot_gen:
                a = _age_seconds(boot_gen)
                boot_stale = a is not None and a > _STALE_SECONDS
            newest_day = ""
            for c in cards:
                d = _read_frontmatter_value(c, "day") or ""
                if d > newest_day:
                    newest_day = d
            agents.append({
                "agent": adir.name,
                "session_cards": len(cards),
                "newest_day": newest_day,
                "boot_pack": boot.is_file(),
                "boot_pack_stale": boot_stale,
                "boot_pack_generated_at": boot_gen or "",
            })

    # pipeline stages
    session_files = [c for c in session_root.rglob("*.md") if c.name != "_BOOT_PACK.md"] if session_root.is_dir() else []
    daily_files = list(daily_root.rglob("*.md")) if daily_root.is_dir() else []
    digest_files = list(digest_root.rglob("*.md")) if digest_root.is_dir() else []
    boot_files = list(session_root.rglob("_BOOT_PACK.md")) if session_root.is_dir() else []

    stages = [
        _stage_from_files("session cards", session_files),
        _stage_from_files("daily roll-ups", daily_files),
        _stage_from_files("digest nodes", digest_files),
        _stage_from_files("boot packs", boot_files),
    ]
    # card index stage (special: a single JSONL file)
    stages.append(_card_index_stage())

    # digests by kind/slug (link targets)
    digests: list[dict] = []
    if digest_root.is_dir():
        for kdir in sorted(p for p in digest_root.iterdir() if p.is_dir()):
            for node in sorted(kdir.glob("*.md")):
                digests.append({
                    "kind": kdir.name,
                    "slug": node.stem,
                    "wikilink": f"{kdir.name}/{node.stem}",
                    "freshness": _read_frontmatter_value(node, "freshness") or "unknown",
                    "source_through": _read_frontmatter_value(node, "source_through") or "",
                })

    return Inventory(layout_present=True, agents=agents, stages=stages, digests=digests)


def _card_index_stage() -> StageState:
    idx = cmp.metadata_dir("indexes", ensure=False) / "card_index.jsonl"
    if not idx.is_file():
        return StageState(name="card index", present=False)
    try:
        first = idx.read_text(encoding="utf-8", errors="ignore").splitlines()[:1]
        meta = json.loads(first[0]).get("_index_meta", {}) if first else {}
    except Exception:  # noqa: BLE001
        meta = {}
    records = meta.get("records", 0)
    gen = meta.get("generated_at", "")
    age = _age_seconds(gen)
    return StageState(name="card index", present=records > 0, count=records, newest=gen,
                      stale=(age is not None and age > _STALE_SECONDS),
                      detail=", ".join(f"{k}={v}" for k, v in sorted(meta.get("by_kind", {}).items())))


# ---------------------------------------------------------------------------
# Build + render
# ---------------------------------------------------------------------------
def build_dashboard(*, cached: bool = False) -> dict:
    health, notes = load_health(cached=cached)
    inv = gather_inventory()
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "health": health,
        "notes": notes,
        "inventory": {
            "layout_present": inv.layout_present,
            "agents": inv.agents,
            "stages": [vars(s) for s in inv.stages],
            "digests": inv.digests,
        },
    }


def _problems_first(checks: list[dict]) -> list[dict]:
    return sorted(checks, key=lambda c: -_SEVERITY.get(c.get("status", "ok"), 0))


def render_markdown(data: dict) -> str:
    h = data["health"]
    inv = data["inventory"]
    checks = h.get("checks", [])
    overall = h.get("overall", "unknown")
    lines: list[str] = []
    lines.append("# 🧠 Memory Control Panel")
    lines.append("")
    glyph = _GLYPH.get(overall, "❔")
    lines.append(f"**Overall health: {glyph} {overall.upper()}**  ·  "
                 f"dashboard generated {data['generated_at']}  ·  "
                 f"health as of {h.get('generated_at', '?')}")
    for n in data.get("notes", []):
        lines.append(f"> ⚠️ {n}")
    lines.append("")

    if not inv["layout_present"]:
        lines.append("_Canonical memory layout is not scaffolded — nothing to report yet._")
        lines.append("")
        return "\n".join(lines) + "\n"

    # Problems first
    problems = [c for c in checks if c.get("status") in ("fail", "warn")]
    if problems:
        lines.append("## ⚠️ Needs attention")
        for c in _problems_first(problems):
            g = _GLYPH.get(c["status"], "❔")
            lines.append(f"- {g} **{c['name']}** — {c.get('detail', '')}")
        lines.append("")

    # Pipeline freshness
    lines.append("## 🔄 Pipeline freshness")
    lines.append("_session → daily → digest → boot pack → card index_")
    lines.append("")
    smap = {"fresh": "🟢 fresh", "stale": "🟡 stale", "missing": "🔴 missing"}
    for s in inv["stages"]:
        st = ("missing" if (not s["present"] or s["count"] == 0)
              else ("stale" if s["stale"] else "fresh"))
        extra = f" · {s['detail']}" if s.get("detail") else ""
        newest = f" · newest {s['newest']}" if s.get("newest") else ""
        lines.append(f"- {smap[st]} — **{s['name']}**: {s['count']} item(s){newest}{extra}")
    lines.append("")

    # What I know — agents
    lines.append("## 👤 Agents")
    if inv["agents"]:
        for a in inv["agents"]:
            bp = ("🟢" if a["boot_pack"] and not a["boot_pack_stale"]
                  else ("🟡 stale" if a["boot_pack"] else "🔴 missing"))
            lines.append(f"- **{a['agent']}** — {a['session_cards']} session card(s), "
                         f"newest day {a['newest_day'] or '—'}, boot pack {bp}")
    else:
        lines.append("- _none yet_")
    lines.append("")

    # What I know — digests by topic (link targets)
    lines.append("## 📚 What I know (digests by topic)")
    if inv["digests"]:
        by_kind: dict[str, list[dict]] = {}
        for d in inv["digests"]:
            by_kind.setdefault(d["kind"], []).append(d)
        for kind in sorted(by_kind):
            lines.append(f"- **{kind}**")
            for d in by_kind[kind]:
                fresh = "" if d["freshness"] == "fresh" else f" _({d['freshness']})_"
                through = f" · through {d['source_through']}" if d["source_through"] else ""
                lines.append(f"    - [[{d['wikilink']}]]{fresh}{through}")
    else:
        lines.append("- _no digest nodes yet (run consolidate_memory_digest.py)_")
    lines.append("")

    # Full health roster
    lines.append("## ✅ Health checks")
    for c in checks:
        g = _GLYPH.get(c.get("status", "skip"), "⚪")
        lines.append(f"- {g} **{c['name']}** — {c.get('detail', '')}")
    lines.append("")

    # Try footer
    lines.append("## ▶️ Try")
    lines.append("```")
    lines.append("# search the readable memory")
    lines.append('python search_memory_cards.py --query "<topic>" --top-k 5')
    lines.append("# rebuild the pipeline")
    lines.append("python generate_daily_session_card.py --agent <a> --transcript <f>")
    lines.append("python consolidate_memory_digest.py --all-agents")
    lines.append("python card_search_index.py")
    lines.append("python generate_memory_dashboard.py")
    lines.append("```")
    lines.append("")
    return "\n".join(lines) + "\n"


def write_dashboard(data: dict, *, out_dir: Path | None = None) -> tuple[Path, Path]:
    d = out_dir or cmp.metadata_dir("catalogs")
    md_path = d / DASHBOARD_MD
    json_path = d / DASHBOARD_JSON
    md_path.write_text(render_markdown(data), encoding="utf-8")
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    return md_path, json_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate the memory control panel dashboard.")
    ap.add_argument("--cached", action="store_true",
                    help="use the last written health JSON instead of re-running checks")
    ap.add_argument("--json", action="store_true", help="print the dashboard JSON to stdout")
    ap.add_argument("--out", help="override output directory")
    args = ap.parse_args(argv)

    data = build_dashboard(cached=args.cached)
    if args.json:
        print(json.dumps(data, ensure_ascii=False, sort_keys=True))
        return 0
    out_dir = Path(args.out) if args.out else None
    md_path, json_path = write_dashboard(data, out_dir=out_dir)
    overall = data["health"].get("overall", "unknown")
    print(f"dashboard: overall={overall} -> {md_path}")
    print(f"           json    -> {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
