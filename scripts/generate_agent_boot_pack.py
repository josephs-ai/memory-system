"""
generate_agent_boot_pack.py — Phase 3 of the ReMe-inspired memory upgrade.

Assemble a single, bounded, provenance-stamped ORIENTATION file per agent — the
"Boot Pack" — so an agent that wakes up fresh has exactly ONE file to read that
tells it: what's been happening (system timeline), what THIS agent did recently
(Phase 2 session-card roll-up), and the active project context.

Design constraints honored (master plan):
- Generated, never hand-maintained.
- Provenance + freshness header on every artifact (generated_at, freshness, sources).
- DEGRADED must be LOUD: a degraded pack says so at the top so the agent announces
  it instead of silently assuming "no recent history".
- Reuse existing seams: timeline daily/latest files, canonical_memory_paths,
  project_memory_paths. No OpenClaw runtime/boot changes (that's the handoff).
- NO LLM in the loop: pure assembly of already-generated, deterministic inputs.

Output: canonical/session/<agent>/_BOOT_PACK.md
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

WORKSPACE = Path.home() / ".openclaw" / "workspace"
TIMELINE_DIR = WORKSPACE / ".memory-index" / "timeline"
TIMELINE_DAILY = TIMELINE_DIR / "daily"

FRESH, STALE, DEGRADED = "fresh", "stale", "degraded"

# Bound the boot pack so it stays an ORIENTATION, not a dump. Timeline daily files
# can be >100KB; an over-long boot pack defeats the purpose and burns context.
MAX_TIMELINE_CHARS = 8000
MAX_CARDS_CHARS = 12000
MAX_PROJECT_CHARS = 6000


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _tail_clip(text: str, limit: int) -> tuple[str, bool]:
    """Keep the most recent (tail) portion of a long doc, since recency matters
    most for orientation. Returns (clipped_text, was_clipped)."""
    text = text.rstrip()
    if len(text) <= limit:
        return text, False
    clipped = text[-limit:]
    # don't cut mid-line at the top
    nl = clipped.find("\n")
    if nl != -1:
        clipped = clipped[nl + 1:]
    return clipped, True


# ---------------------------------------------------------------------------
# Source resolution
# ---------------------------------------------------------------------------
def _latest_timeline_daily() -> Path | None:
    if not TIMELINE_DAILY.is_dir():
        return None
    files = sorted(p for p in TIMELINE_DAILY.glob("*.md")
                   if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}\.md", p.name))
    return files[-1] if files else None


def _read(path: Path | None) -> str:
    if not path or not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return ""


@dataclass
class BootSource:
    name: str
    path: str | None
    ok: bool
    chars: int = 0
    clipped: bool = False
    note: str = ""


@dataclass
class BootPack:
    agent: str
    generated_at: str
    freshness: str
    timeline_day: str | None = None
    cards_day: str | None = None
    cards_stale: bool = False
    project_id: str | None = None
    sources: list[BootSource] = field(default_factory=list)
    timeline_text: str = ""
    cards_text: str = ""
    project_text: str = ""
    degraded_reasons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_boot_pack(agent: str, project_id: str | None = None) -> BootPack:
    agent = cmp.sanitize_agent(agent)
    gen = _now_iso()
    sources: list[BootSource] = []
    degraded: list[str] = []

    # --- 1. System timeline (global context) ---------------------------------
    tl_path = _latest_timeline_daily()
    timeline_day = tl_path.stem if tl_path else None
    tl_raw = _read(tl_path)
    if not tl_raw:
        # fall back to latest.md if daily missing
        latest = TIMELINE_DIR / "latest.md"
        tl_raw = _read(latest)
        if tl_raw:
            tl_path = latest
        else:
            degraded.append("system timeline missing (no daily file and no latest.md)")
    tl_text, tl_clip = _tail_clip(tl_raw, MAX_TIMELINE_CHARS) if tl_raw else ("", False)
    sources.append(BootSource(
        name="system_timeline", path=str(tl_path) if tl_path else None,
        ok=bool(tl_raw), chars=len(tl_text), clipped=tl_clip,
        note="" if tl_raw else "no timeline content",
    ))

    # --- 2. This agent's recent session cards (Phase 2 roll-up) ---------------
    cards_text = ""
    cards_clip = False
    cards_path: Path | None = None
    cards_day: str | None = None
    cards_stale = False
    reference_day = timeline_day or cmp.today_utc()
    if cmp.layout_present():
        # prefer the roll-up for the reference (timeline) day; else newest across days
        candidate = cmp.daily_agent_card(agent, reference_day, ensure=False)
        if candidate.exists():
            cards_day = reference_day
        else:
            # newest available daily roll-up for this agent across days
            base = cmp.canonical_root() / "daily"
            if base.is_dir():
                for ddir in sorted(base.iterdir(), reverse=True):
                    c = ddir / f"{agent}.md"
                    if c.exists():
                        candidate = c
                        # the day is the parent dir name (YYYY-MM-DD)
                        cards_day = ddir.name if re.fullmatch(
                            r"[0-9]{4}-[0-9]{2}-[0-9]{2}", ddir.name) else None
                        break
        cards_path = candidate
        cards_raw = _read(candidate)
        if cards_raw:
            cards_text, cards_clip = _tail_clip(cards_raw, MAX_CARDS_CHARS)
            # Honesty: a card older than the reference day is STALE, not "Recent".
            if cards_day and cards_day < reference_day:
                cards_stale = True
                degraded.append(
                    f"session cards are stale: newest roll-up is {cards_day}, "
                    f"but current activity is {reference_day}")
        else:
            degraded.append(f"no session-card roll-up for agent '{agent}'")
    else:
        degraded.append("canonical layout not scaffolded (no session cards available)")
    cards_note = ""
    if not cards_text:
        cards_note = "no session-card roll-up"
    elif cards_stale:
        cards_note = f"STALE (from {cards_day}, older than {reference_day})"
    sources.append(BootSource(
        name="session_cards", path=str(cards_path) if cards_path else None,
        # A stale card is NOT a healthy source: count it as not-ok so freshness
        # and provenance reflect that the agent has no CURRENT activity record.
        ok=bool(cards_text) and not cards_stale, chars=len(cards_text),
        clipped=cards_clip, note=cards_note,
    ))

    # --- 3. Project context (optional) ---------------------------------------
    project_text = ""
    proj_clip = False
    proj_path: Path | None = None
    if project_id:
        try:
            import project_memory_paths as pmp

            pid = pmp.normalize_project_id(project_id)
            cur = pmp.get_project_current_file(pid)
            proj_path = cur
            proj_raw = _read(cur)
            if not proj_raw:
                latest_daily = pmp.get_latest_project_daily_file(pid)
                proj_raw = _read(latest_daily)
                if proj_raw:
                    proj_path = latest_daily
            if proj_raw:
                project_text, proj_clip = _tail_clip(proj_raw, MAX_PROJECT_CHARS)
            else:
                degraded.append(f"project '{project_id}' has no current/daily memory")
        except Exception as e:  # noqa: BLE001
            degraded.append(f"project context unavailable: {e}")
        sources.append(BootSource(
            name="project_context", path=str(proj_path) if proj_path else None,
            ok=bool(project_text), chars=len(project_text), clipped=proj_clip,
            note="" if project_text else "no project memory",
        ))

    # --- Freshness verdict ---------------------------------------------------
    # DEGRADED if we have no CURRENT orientation source: no timeline AND no fresh
    # cards (a stale card doesn't count as current — it would mislead). We still
    # render the stale card content, but clearly marked and not as a freshness pass.
    have_current_cards = bool(cards_text) and not cards_stale
    if not tl_text and not have_current_cards:
        freshness = DEGRADED
        if not degraded:
            degraded.append("no timeline and no current session cards available")
    else:
        freshness = FRESH

    return BootPack(
        agent=agent, generated_at=gen, freshness=freshness,
        timeline_day=timeline_day, cards_day=cards_day, cards_stale=cards_stale,
        project_id=project_id, sources=sources, timeline_text=tl_text,
        cards_text=cards_text, project_text=project_text, degraded_reasons=degraded,
    )


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
def render_boot_pack(p: BootPack) -> str:
    fm = [
        "---",
        f"agent: {p.agent}",
        f"generated_at: {p.generated_at}",
        f"freshness: {p.freshness}",
        f"timeline_day: {p.timeline_day or '?'}",
        f"cards_day: {p.cards_day or '-'}",
        f"cards_stale: {'true' if p.cards_stale else 'false'}",
        f"project_id: {p.project_id or '-'}",
        f"sources_ok: {sum(1 for s in p.sources if s.ok)}/{len(p.sources)}",
        "---",
        "",
        f"# Boot Pack: {p.agent}",
        f"> Generated {p.generated_at} — read this FIRST to orient yourself.",
        "",
    ]
    body: list[str] = []

    if p.freshness == DEGRADED:
        body += [
            "> **⚠️ DEGRADED MEMORY — DO NOT GUESS.**",
            "> The orientation sources below are missing or unreadable. Announce this",
            "> degraded state and inspect raw transcripts before asserting recent history.",
            "",
        ]
    if p.degraded_reasons:
        body.append("> Gaps detected:")
        for r in p.degraded_reasons:
            body.append(f"> - {r}")
        body.append("")

    # 1. timeline
    body.append("## 1. System Timeline (global context)")
    if p.timeline_text:
        if any(s.name == "system_timeline" and s.clipped for s in p.sources):
            body.append("_(showing most recent portion)_")
        body += ["", p.timeline_text, ""]
    else:
        body += ["_No system timeline available — this is a gap._", ""]

    # 2. cards
    if p.cards_stale:
        body.append(f"## 2. Your Activity — ⚠️ STALE (from {p.cards_day}, not current)")
    else:
        body.append("## 2. Your Recent Activity (session cards)")
    if p.cards_text:
        if p.cards_stale:
            body.append(
                f"> **⚠️ No session cards for the current period.** The roll-up below "
                f"is from {p.cards_day} and may be out of date — do not treat it as "
                f"the latest state; verify against the timeline / raw transcripts.")
        if any(s.name == "session_cards" and s.clipped for s in p.sources):
            body.append("_(showing most recent portion)_")
        body += ["", p.cards_text, ""]
    else:
        body += ["_No session-card roll-up for you yet — run the daily-card generator._", ""]

    # 3. project
    if p.project_id:
        body.append("## 3. Project Context")
        if p.project_text:
            body += ["", p.project_text, ""]
        else:
            body += ["_No project memory found._", ""]

    # provenance
    body.append("## Provenance")
    for s in p.sources:
        flag = "ok" if s.ok else "MISSING"
        clip = " (clipped)" if s.clipped else ""
        note = f" [{s.note}]" if s.note else ""
        body.append(f"- {s.name}: {flag}{clip}{note} — {s.path or '-'} ({s.chars} chars)")
    body.append("")

    return "\n".join(fm + body).rstrip() + "\n"


def boot_pack_path(agent: str) -> Path:
    agent = cmp.sanitize_agent(agent)
    return cmp.session_dir(agent) / "_BOOT_PACK.md"


def write_boot_pack(p: BootPack) -> Path:
    out = boot_pack_path(p.agent)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_boot_pack(p), encoding="utf-8")
    return out


def generate_one(agent: str, project_id: str | None = None) -> tuple[Path, BootPack]:
    cmp.scaffold(write_readme=True)
    pack = build_boot_pack(agent, project_id=project_id)
    return write_boot_pack(pack), pack


def _known_agents() -> list[str]:
    agents_dir = Path.home() / ".openclaw" / "agents"
    out: list[str] = []
    if agents_dir.is_dir():
        for d in sorted(agents_dir.iterdir()):
            if (d / "sessions").is_dir():
                out.append(d.name)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate per-agent memory boot packs")
    ap.add_argument("--agent")
    ap.add_argument("--project")
    ap.add_argument("--all-agents", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.all_agents:
        agents = _known_agents()
    elif args.agent:
        agents = [args.agent]
    else:
        ap.error("provide --agent NAME or --all-agents")

    results = []
    for agent in agents:
        path, pack = generate_one(agent, project_id=args.project)
        results.append({
            "agent": pack.agent,
            "boot_pack": str(path),
            "freshness": pack.freshness,
            "sources_ok": sum(1 for s in pack.sources if s.ok),
            "sources": len(pack.sources),
            "degraded_reasons": pack.degraded_reasons,
        })

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            print(f"{r['agent']}: {r['freshness']} "
                  f"(sources {r['sources_ok']}/{r['sources']}) -> {r['boot_pack']}")
            for reason in r["degraded_reasons"]:
                print(f"    gap: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
