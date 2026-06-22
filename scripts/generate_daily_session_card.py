"""
generate_daily_session_card.py — Phase 2 "auto_memory".

Generate a concise, structured, provenance-stamped DAILY SESSION CARD from a real
transcript, written into the Phase-1 canonical tree. This is the readable per-session
layer that was missing when an agent couldn't reconstruct WHY something happened
(the raw facts were in the transcript, but no card carried
summary/decisions/files/tests/blockers/next-step).

NO LLM in the loop: extraction is deterministic and unit-testable, matching the
repo's "no model in the retrieval/derivation path" ethos. Reuses
timeline_event_ledger for transcript parsing/text extraction so cards stay
consistent with the timeline, and canonical_memory_paths for layout.

Outputs:
  canonical/session/<agent>/<session-id>.md     per-session card (durable)
  canonical/daily/<day>/<agent>.md               per-agent day roll-up (newest cards)
  canonical/daily/<day>/_index.md                cross-agent day roll-up
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
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import canonical_memory_paths as cmp
import timeline_event_ledger as tel

OPENCLAW_ROOT = Path(__file__).resolve().parents[3] if False else (Path.home() / ".openclaw")
AGENTS_DIR = OPENCLAW_ROOT / "agents"

FRESH, STALE, DEGRADED = "fresh", "stale", "degraded"

# ---------------------------------------------------------------------------
# Extraction heuristics (transparent, testable, no model dependency)
# ---------------------------------------------------------------------------
_COMMIT_MARKERS = ("i'll ", "i will ", "done.", "done ", "committed", "fixed",
                    "shipped", "implemented", "added ", "wired ", "created ")
_BLOCKER_RE = re.compile(
    r"\b(blocked|cannot|can't|cant|failed|failure|error|timed out|timeout|refused|"
    r"missing|broken|stale|conflict)\b",
    re.IGNORECASE,
)
_NEXT_RE = re.compile(r"(want me to|next step|shall i|should i|do you want|standing by|"
                      r"on approval|let me know)", re.IGNORECASE)
# Only match commands that actually INVOKE a test runner, not any command that
# merely mentions a test_*.py path (e.g. `git log` over a branch named test_...)
# nor a command that only *talks about* pytest (echo/grep/cat/comments).
_TEST_RE = re.compile(r"(\bpytest\b|\bunittest\b|-m\s+pytest|python[0-9.]*\s+-m\s+pytest)", re.IGNORECASE)
# A segment is a "talking about" segment if it merely prints/searches/comments.
_NON_RUNNER_HEAD = re.compile(r"^(#|(echo|printf|grep|rg|cat|less|sed|awk|head|tail)\b)", re.IGNORECASE)


def _command_invokes_tests(cmd: str) -> bool:
    """True only if a shell segment actually runs a test runner. Splits on shell
    separators so `echo x && pytest ...` still counts, but `echo 'run pytest'`,
    `grep pytest setup.cfg`, and `# pytest is great` do not."""
    # Split on ; && || | and newlines into individual command segments.
    for seg in re.split(r"(?:&&|\|\||[;|\n])", cmd):
        seg = seg.strip()
        if not seg or _NON_RUNNER_HEAD.match(seg):
            continue
        # Strip a leading `cd ... &&`-style prefix already handled by split; check
        # the segment head token chain for an actual runner invocation.
        if _TEST_RE.search(seg):
            return True
    return False
_PASS_TAIL_RE = re.compile(r"(\d+\s+passed[^\n]*|\d+\s+failed[^\n]*|\bFAILED\b[^\n]*)")


def _iter_message_blocks(events: list[dict[str, Any]]):
    """Yield (role, block) for every content block in message events."""
    for ev in events:
        if ev.get("type") != "message":
            continue
        msg = ev.get("message") or {}
        role = str(msg.get("role") or "")
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    yield role, block


def _tool_calls(events: list[dict[str, Any]]):
    for role, block in _iter_message_blocks(events):
        if block.get("type") in ("toolCall", "tool_use"):
            yield block


def extract_files_changed(events: list[dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    for tc in _tool_calls(events):
        if tc.get("name") in ("edit", "write"):
            args = tc.get("arguments") or tc.get("input") or {}
            path = args.get("file_path") or args.get("path")
            if path and path not in seen:
                seen.append(str(path))
    return seen


def extract_tests_run(events: list[dict[str, Any]]) -> list[str]:
    cmds: list[str] = []
    for tc in _tool_calls(events):
        if tc.get("name") == "exec":
            args = tc.get("arguments") or tc.get("input") or {}
            cmd = str(args.get("command") or "")
            if _command_invokes_tests(cmd):
                first = " ".join(cmd.split())[:160]
                if first not in cmds:
                    cmds.append(first)
    return cmds


def _assistant_texts(events: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for role, block in _iter_message_blocks(events):
        if role == "assistant" and block.get("type") == "text":
            t = str(block.get("text") or "").strip()
            if t:
                out.append(t)
    return out


def _user_texts(events: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for role, block in _iter_message_blocks(events):
        if role == "user" and block.get("type") == "text":
            t = str(block.get("text") or "").strip()
            if t and t != "HEARTBEAT_OK":
                out.append(t)
    return out


def _clean_user_directive(text: str) -> str:
    """Strip the untrusted-metadata JSON envelope + timestamps that wrap webchat
    user messages, leaving the actual instruction."""
    # remove ```json ... ``` blocks
    text = re.sub(r"```json.*?```", "", text, flags=re.DOTALL)
    # remove a leading [Day YYYY-MM-DD HH:MM TZ] stamp
    text = re.sub(r"\[\w{3}\s+\d{4}-\d{2}-\d{2}[^\]]*\]", "", text)
    text = re.sub(r"Sender \(untrusted metadata\):", "", text)
    return " ".join(text.split()).strip()


def _strip_md(text: str) -> str:
    """Strip markdown emphasis/list noise from an inline snippet."""
    return " ".join(text.replace("**", "").replace("*", "").split()).strip(" -:")


def _starts_commitment(text: str) -> bool:
    """True only when a commitment marker is at the START of the (md-stripped) text.
    Prevents mis-tagging analytical sentences like 'This added complexity is...'
    as a commitment just because they contain a marker word mid-sentence."""
    head = _strip_md(text).lower()
    return any(head.startswith(m) for m in _COMMIT_MARKERS)


def extract_decisions(events: list[dict[str, Any]], *, cap: int = 10) -> list[str]:
    # Collect both kinds separately so a chatty user can't starve the assistant
    # commitments (or vice-versa); then merge with a balanced share of the cap.
    user_items: list[str] = []
    seen_user: set[str] = set()
    for t in _user_texts(events):
        c = _clean_user_directive(t)
        if not c:
            continue
        key = c.lower()
        if len(c) <= 200 and key not in seen_user:
            seen_user.add(key)
            user_items.append(f"(user) {c}")

    did_items: list[str] = []
    for t in _assistant_texts(events):
        first_line = t.strip().splitlines()[0] if t.strip() else ""
        if _starts_commitment(t) or _starts_commitment(first_line):
            tag = f"(did) {_strip_md(t)[:200]}"
            if tag not in did_items:
                did_items.append(tag)

    # Reserve up to half the cap for each side; backfill from whichever has more.
    half = cap // 2
    merged = user_items[:half] + did_items[:half]
    if len(merged) < cap:
        merged += user_items[half:half + (cap - len(merged))]
    if len(merged) < cap:
        merged += did_items[half:half + (cap - len(merged))]
    return merged[:cap]


def extract_blockers(events: list[dict[str, Any]], *, cap: int = 6) -> list[str]:
    blockers: list[str] = []
    for t in _assistant_texts(events):
        for line in t.splitlines():
            line = line.strip(" -*")
            if len(line) < 8:
                continue
            if _BLOCKER_RE.search(line):
                s = _strip_md(line)[:200]
                if s and s not in blockers:
                    blockers.append(s)
    return blockers[:cap]


def extract_next_step(events: list[dict[str, Any]]) -> str | None:
    texts = _assistant_texts(events)
    for t in reversed(texts):
        if _NEXT_RE.search(t):
            # take the sentence containing the marker
            for sent in re.split(r"(?<=[.?!])\s+", t):
                if _NEXT_RE.search(sent):
                    return " ".join(sent.split())[:240]
    return None


def extract_summary(events: list[dict[str, Any]], *, cap: int = 6) -> list[str]:
    """Salient assistant lines: commitments + markdown headers, deduped, bounded."""
    out: list[str] = []
    for t in _assistant_texts(events):
        for line in t.splitlines():
            s = line.strip()
            is_header = s.startswith("#") or s.startswith("**")
            is_commit = _starts_commitment(s)
            if (is_header or is_commit) and len(s) > 6:
                clean = _strip_md(s.lstrip("#"))[:180]
                if clean and clean not in out:
                    out.append(clean)
            if len(out) >= cap:
                return out
    return out


# ---------------------------------------------------------------------------
# Card model
# ---------------------------------------------------------------------------
@dataclass
class CardModel:
    agent: str
    session_id: str
    transcript_path: str
    generated_at: str
    freshness: str
    event_count: int = 0
    span_start: str | None = None
    span_end: str | None = None
    day: str | None = None
    summary: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    files_changed: list[str] = field(default_factory=list)
    tests_run: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    next_step: str | None = None
    note: str | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _session_id_from_path(path: Path) -> str:
    name = path.name
    for suffix in (".jsonl",):
        if suffix in name:
            name = name.split(suffix)[0]
    return name or path.stem


def build_card(agent: str, transcript_path: str | Path, *, day: str | None = None) -> CardModel:
    path = Path(transcript_path)
    sid = _session_id_from_path(path)
    gen = _now_iso()

    if not path.exists():
        return CardModel(
            agent=agent, session_id=sid, transcript_path=str(path),
            generated_at=gen, freshness=DEGRADED,
            note="transcript not found — card is a degraded placeholder",
            day=day or cmp.today_utc(),
        )
    try:
        events = tel.load_session_events(path)
    except Exception as e:  # noqa: BLE001
        return CardModel(
            agent=agent, session_id=sid, transcript_path=str(path),
            generated_at=gen, freshness=DEGRADED,
            note=f"transcript unreadable: {e}", day=day or cmp.today_utc(),
        )

    msg_events = [e for e in events if e.get("type") == "message"]
    if not msg_events:
        return CardModel(
            agent=agent, session_id=sid, transcript_path=str(path),
            generated_at=gen, freshness=DEGRADED,
            note="transcript has no message events", event_count=0,
            day=day or cmp.today_utc(),
        )

    timestamps = [str(e.get("timestamp") or "") for e in msg_events if e.get("timestamp")]
    span_start = min(timestamps) if timestamps else None
    span_end = max(timestamps) if timestamps else None
    derived_day = day or ((span_end or span_start or "")[:10] or cmp.today_utc())
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", derived_day):
        derived_day = cmp.today_utc()

    return CardModel(
        agent=agent,
        session_id=sid,
        transcript_path=str(path),
        generated_at=gen,
        freshness=FRESH,
        event_count=len(msg_events),
        span_start=span_start,
        span_end=span_end,
        day=derived_day,
        summary=extract_summary(events),
        decisions=extract_decisions(events),
        files_changed=extract_files_changed(events),
        tests_run=extract_tests_run(events),
        blockers=extract_blockers(events),
        next_step=extract_next_step(events),
    )


def render_card(m: CardModel) -> str:
    fm = [
        "---",
        f"agent: {m.agent}",
        f"session_id: {m.session_id}",
        f"generated_at: {m.generated_at}",
        f"source_transcript: {m.transcript_path}",
        f"source_event_count: {m.event_count}",
        f"span: {m.span_start or '?'} .. {m.span_end or '?'}",
        f"day: {m.day or '?'}",
        f"freshness: {m.freshness}",
        "---",
        "",
        f"# Session Card: {m.agent} — {m.day or '?'}",
        "",
    ]
    body: list[str] = []
    if m.note:
        body += [f"> NOTE: {m.note}", ""]

    def section(title: str, items: list[str]) -> None:
        if not items:
            return
        body.append(f"## {title}")
        for it in items:
            body.append(f"- {it}")
        body.append("")

    section("Summary", m.summary)
    section("Decisions", m.decisions)
    section("Files changed", m.files_changed)
    section("Tests run", m.tests_run)
    section("Blockers", m.blockers)
    if m.next_step:
        body += ["## Next step", f"- {m.next_step}", ""]
    body += [
        "## Provenance",
        f"- transcript: {m.transcript_path}",
        f"- events: {m.event_count}",
        f"- span: {m.span_start or '?'} .. {m.span_end or '?'}",
        "",
    ]
    return "\n".join(fm + body).rstrip() + "\n"


def write_card(m: CardModel) -> Path:
    out = cmp.session_card(m.agent, m.session_id)
    out.write_text(render_card(m), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Day roll-ups
# ---------------------------------------------------------------------------
def _read_card_meta(card_path: Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    try:
        lines = card_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return meta
    if lines and lines[0].strip() == "---":
        for line in lines[1:]:
            if line.strip() == "---":
                break
            if ":" in line:
                k, _, v = line.partition(":")
                meta[k.strip()] = v.strip()
    return meta


def rebuild_agent_daily(agent: str, day: str) -> Path:
    """Regenerate canonical/daily/<day>/<agent>.md from that agent's session cards
    whose frontmatter day == day. Newest first."""
    sess_dir = cmp.session_dir(agent)
    matches: list[tuple[str, Path]] = []
    for card in sess_dir.glob("*.md"):
        meta = _read_card_meta(card)
        if meta.get("day") == day:
            matches.append((meta.get("generated_at", ""), card))
    matches.sort(reverse=True)

    out = cmp.daily_agent_card(agent, day)
    lines = [f"# Daily roll-up: {agent} — {day}", "",
             f"_Generated {_now_iso()} from {len(matches)} session card(s)._", ""]
    if not matches:
        lines += ["_No session cards for this day._", ""]
    for _, card in matches:
        meta = _read_card_meta(card)
        lines.append(f"## Session {meta.get('session_id', card.stem)}  "
                     f"(freshness: {meta.get('freshness', '?')})")
        # inline the card body (skip its frontmatter)
        text = card.read_text(encoding="utf-8", errors="ignore")
        parts = text.split("---", 2)
        body = parts[2] if len(parts) >= 3 else text
        lines.append(body.strip())
        lines.append("")
    out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return out


def rebuild_day_index(day: str) -> Path:
    """Cross-agent index of all per-agent roll-ups present for the day."""
    ddir = cmp.daily_dir(day)
    agent_files = sorted(p for p in ddir.glob("*.md") if p.name != "_index.md")
    lines = [f"# Day index — {day}", "",
             f"_Generated {_now_iso()}. {len(agent_files)} agent(s) active._", ""]
    for af in agent_files:
        agent = af.stem
        meta_count = af.read_text(encoding="utf-8", errors="ignore").count("## Session ")
        lines.append(f"- **{agent}** — {meta_count} session(s) → `daily/{day}/{af.name}`")
    lines.append("")
    out = cmp.daily_index(day)
    out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Batch discovery
# ---------------------------------------------------------------------------
def recent_transcripts(*, since_days: float) -> list[tuple[str, Path]]:
    cutoff = time.time() - since_days * 86400
    rows: list[tuple[str, Path]] = []
    if not AGENTS_DIR.exists():
        return rows
    for agent_dir in sorted(AGENTS_DIR.iterdir()):
        sdir = agent_dir / "sessions"
        if not sdir.is_dir():
            continue
        for pattern in ("*.jsonl", "*.jsonl.reset.*", "*.jsonl.deleted.*"):
            for p in sdir.glob(pattern):
                try:
                    if p.stat().st_mtime < cutoff:
                        continue
                except OSError:
                    continue
                rows.append((agent_dir.name, p))
    return rows


def generate_one(agent: str, transcript: Path, *, day: str | None = None) -> tuple[Path, CardModel]:
    model = build_card(agent, transcript, day=day)
    card_path = write_card(model)
    if model.day:
        rebuild_agent_daily(agent, model.day)
        rebuild_day_index(model.day)
    return card_path, model


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate daily session cards (auto_memory)")
    ap.add_argument("--agent")
    ap.add_argument("--transcript")
    ap.add_argument("--all-agents", action="store_true")
    ap.add_argument("--since-days", type=float, default=1.0)
    ap.add_argument("--day")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    cmp.scaffold(write_readme=True)  # ensure layout exists; idempotent

    jobs: list[tuple[str, Path]] = []
    if args.all_agents:
        jobs = recent_transcripts(since_days=args.since_days)
    elif args.transcript and args.agent:
        jobs = [(args.agent, Path(args.transcript))]
    elif args.agent:
        latest = tel.find_latest_session_for_agent(args.agent)
        if not latest:
            print(f"No session found for agent {args.agent}", file=sys.stderr)
            return 1
        jobs = [(args.agent, latest)]
    else:
        ap.error("provide --all-agents, or --agent (+optional --transcript)")

    results = []
    for agent, transcript in jobs:
        if args.dry_run:
            results.append({"agent": agent, "transcript": str(transcript), "action": "would-generate"})
            continue
        card_path, model = generate_one(agent, transcript, day=args.day)
        results.append({
            "agent": agent,
            "session_id": model.session_id,
            "card": str(card_path),
            "freshness": model.freshness,
            "files_changed": len(model.files_changed),
            "tests_run": len(model.tests_run),
            "decisions": len(model.decisions),
            "blockers": len(model.blockers),
        })

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            if args.dry_run:
                print(f"[dry-run] {r['agent']}: {r['transcript']}")
            else:
                print(f"{r['agent']}/{r['session_id']}: {r['freshness']} "
                      f"(files {r['files_changed']}, tests {r['tests_run']}, "
                      f"decisions {r['decisions']}, blockers {r['blockers']}) -> {r['card']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
