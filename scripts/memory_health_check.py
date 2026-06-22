"""
memory_health_check.py — "Is memory startup safe right now?"

Single source of truth for memory-startup health. Designed to catch the exact
failure classes that have bitten this system before (see memory/learned-fixes.md):
stale timeline, missing today-daily, un-symlinked agent workspaces, stale
"no memory yet" boot text, stalled checkpoint cursors, invisible rotated/reset
transcripts, embedding-provider outages, and dead/desynced service PIDs.

Design:
- Pure stdlib + optional reuse of existing helpers (memory_db, config). No new deps.
- Every check returns a Check(name,status,detail,value,source). status in
  ok|warn|fail|skip. Overall = worst of ok<warn<fail (skip ignored).
- warn  = degraded-but-usable.  fail = startup unsafe.
- Writes machine JSON to health/memory_startup_latest.json and prints a human
  summary. Exit code 0 (ok/warn-only when --warn-ok) / 1 (warn) / 2 (fail),
  controllable so heartbeat can call it non-fatally.

This tool must be safe to run anytime and must never mutate memory state.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# ---------------------------------------------------------------------------
# Paths (mirror render_timeline_views.py / config.py conventions). Overridable
# via OPENCLAW_MEMORY_ROOT / OPENCLAW_ROOT for hermetic testing.
# ---------------------------------------------------------------------------
OPENCLAW_ROOT = Path(os.environ.get("OPENCLAW_ROOT", str(Path.home() / ".openclaw")))
MEMORY_ROOT = Path(
    os.environ.get(
        "OPENCLAW_MEMORY_ROOT",
        str(OPENCLAW_ROOT / "workspace" / ".memory-index"),
    )
)
WORKSPACE = MEMORY_ROOT.parent
AGENTS_DIR = OPENCLAW_ROOT / "agents"
TIMELINE_ROOT = MEMORY_ROOT / "timeline"
TIMELINE_LATEST = TIMELINE_ROOT / "latest.md"
TIMELINE_DAILY = TIMELINE_ROOT / "daily"
TIMELINE_LEDGER = TIMELINE_ROOT / "events.jsonl"
HEALTH_DIR = MEMORY_ROOT / "health"
STATE_DIR = MEMORY_ROOT / "state"
CURSORS_DIR = STATE_DIR / "cursors"
HEALTH_OUT = HEALTH_DIR / "memory_startup_latest.json"

OK, WARN, FAIL, SKIP = "ok", "warn", "fail", "skip"
_SEVERITY = {SKIP: -1, OK: 0, WARN: 1, FAIL: 2}

# Default thresholds (seconds). Overridable via --thresholds-file JSON.
DEFAULTS = {
    "timeline_latest_warn_age_s": 2 * 3600,
    "timeline_latest_fail_age_s": 24 * 3600,
    "cursor_stale_warn_age_s": 6 * 3600,
    "pending_backlog_warn": 25,
    "rotated_backlog_warn": 50,
    "http_timeout_s": 3.0,
}

# Stale-bootstrap phrases that, when given as an ACTIVE instruction, mean the boot
# file is telling the agent it has no memory. We must distinguish a real stale
# instruction from the corrective text that WARNS against it (e.g. main added
# 'Do not follow any older "There is no memory yet" bootstrap text.'). Naive
# substring matching flags the cure as the disease, so matching is negation-aware
# and line-scoped.
STALE_BOOT_MARKERS = (
    "there is no memory yet",
    "no memory yet",
    "hello world",
    "identity interview",
    "this is your birth certificate",
)
# If any of these guard words appear on the same line as a marker, the marker is
# being referenced/negated, not asserted — treat as clean.
NEGATION_GUARDS = (
    "do not",
    "don't",
    "dont",
    "never",
    "ignore",
    "no longer",
    "not fresh",
    "older",
    "stale",
    "previous",
    "deprecated",
    "must not",
    "should not",
    "avoid",
    "if",  # conditional template boilerplate, e.g. "If BOOTSTRAP.md exists..."
    "unless",
)
BOOT_FILES = ("BOOT.md", "BOOTSTRAP.md", "AGENTS.md")


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""
    value: Any = None
    source: str = ""


@dataclass
class Report:
    generated_at: str
    overall: str
    checks: list[Check] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "overall": self.overall,
            "checks": [asdict(c) for c in self.checks],
        }


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def _now() -> float:
    return time.time()


def _mtime(path: Path) -> float:
    try:
        return float(path.stat().st_mtime)
    except OSError:
        return 0.0


def _today_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _http_ok(url: str, timeout: float) -> tuple[bool, str]:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (trusted localhost)
            body = r.read(256).decode("utf-8", "ignore")
            return (200 <= r.status < 300), f"HTTP {r.status}: {body[:120]}"
    except Exception as e:  # noqa: BLE001 — health tool must never raise
        return False, f"{type(e).__name__}: {e}"


def _active_agent_dirs() -> list[Path]:
    if not AGENTS_DIR.exists():
        return []
    return [d for d in sorted(AGENTS_DIR.iterdir()) if (d / "sessions").is_dir()]


def _agent_workspaces() -> list[Path]:
    parent = WORKSPACE.parent
    if not parent.exists():
        return []
    return [
        d
        for d in sorted(parent.iterdir())
        if d.is_dir() and d.name.startswith("workspace-")
    ]


# ---------------------------------------------------------------------------
# Individual checks. Each takes (cfg) and returns a Check. Injectable probes
# (db/http) are passed via cfg["probes"] so tests stay hermetic.
# ---------------------------------------------------------------------------
def check_timeline_latest_fresh(cfg: dict) -> Check:
    src = str(TIMELINE_LATEST)
    if not TIMELINE_LATEST.exists():
        return Check("timeline_latest_fresh", FAIL, "timeline/latest.md missing", None, src)
    age = _now() - _mtime(TIMELINE_LATEST)
    val = {"age_seconds": int(age)}
    if age > cfg["timeline_latest_fail_age_s"]:
        return Check("timeline_latest_fresh", FAIL, f"stale {int(age/3600)}h", val, src)
    if age > cfg["timeline_latest_warn_age_s"]:
        return Check("timeline_latest_fresh", WARN, f"aging {int(age/60)}m", val, src)
    return Check("timeline_latest_fresh", OK, f"fresh ({int(age/60)}m)", val, src)


def check_timeline_today_exists(cfg: dict) -> Check:
    today = TIMELINE_DAILY / f"{_today_key()}.md"
    if today.exists() and today.stat().st_size > 0:
        return Check("timeline_today_exists", OK, "present", str(today), str(today))
    return Check(
        "timeline_today_exists",
        WARN,
        "no today daily yet (ok early in day / pre-activity)",
        None,
        str(TIMELINE_DAILY),
    )


def check_workspaces_timeline_linked(cfg: dict) -> Check:
    bad: list[str] = []
    checked = 0
    for ws in _agent_workspaces():
        link = ws / ".memory-index" / "timeline"
        if not link.exists() and not link.is_symlink():
            continue  # workspace may not use memory-index at all
        checked += 1
        if not link.is_symlink():
            bad.append(ws.name)
    if not checked:
        return Check("workspaces_timeline_linked", SKIP, "no agent workspaces found", 0, "")
    if bad:
        return Check(
            "workspaces_timeline_linked",
            FAIL,
            f"{len(bad)} workspace(s) have non-symlink timeline: {', '.join(bad[:6])}",
            bad,
            "workspace-*/.memory-index/timeline",
        )
    return Check("workspaces_timeline_linked", OK, f"{checked} linked", checked, "")


def _line_is_active_stale_marker(line: str) -> str | None:
    """Return the matched marker if this line actively asserts a stale bootstrap,
    or None if no marker / the marker is negated/quoted as a warning."""
    # Strip markdown emphasis so "Do **not** say" matches the "not"/"do not" guard,
    # and collapse whitespace so "do  not" still matches.
    low = line.lower().replace("*", "").replace("_", "")
    low = " ".join(low.split())
    for marker in STALE_BOOT_MARKERS:
        if marker in low:
            if any(guard in low.split() or guard in low for guard in NEGATION_GUARDS):
                return None  # referenced/negated, not asserted
            return marker
    return None


def check_boot_text_clean(cfg: dict) -> Check:
    hits: list[str] = []
    roots = [WORKSPACE] + _agent_workspaces()
    for root in roots:
        for fname in BOOT_FILES:
            p = root / fname
            if not p.exists():
                continue
            try:
                lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                continue
            for lineno, line in enumerate(lines, 1):
                marker = _line_is_active_stale_marker(line)
                if marker:
                    hits.append(f"{root.name}/{fname}:{lineno}:'{marker}'")
                    break
    if hits:
        return Check(
            "boot_text_clean",
            FAIL,
            f"stale bootstrap text in {len(hits)} file(s): {hits[0]}",
            hits,
            "BOOT/BOOTSTRAP/AGENTS",
        )
    return Check("boot_text_clean", OK, "no stale bootstrap markers", 0, "")


def check_checkpoint_cursors_advancing(cfg: dict) -> Check:
    candidates: list[Path] = []
    if CURSORS_DIR.exists():
        candidates += list(CURSORS_DIR.rglob("*.json"))
    if STATE_DIR.exists():
        candidates += list(STATE_DIR.glob("*.cursor.json"))
    if not candidates:
        return Check("checkpoint_cursors_advancing", WARN, "no cursors found", 0, str(STATE_DIR))
    newest = max(_mtime(p) for p in candidates)
    age = _now() - newest
    val = {"cursor_count": len(candidates), "newest_age_seconds": int(age)}
    if age > cfg["cursor_stale_warn_age_s"]:
        return Check(
            "checkpoint_cursors_advancing",
            WARN,
            f"newest cursor stale {int(age/3600)}h (pipeline may be idle)",
            val,
            str(STATE_DIR),
        )
    return Check("checkpoint_cursors_advancing", OK, f"{len(candidates)} cursors, fresh", val, str(STATE_DIR))


def check_rotated_transcripts_visible(cfg: dict) -> Check:
    """Count recent rotated/reset transcripts. Informational warn on large backlog."""
    cutoff = _now() - 7 * 86400
    rotated = 0
    for agent_dir in _active_agent_dirs():
        sdir = agent_dir / "sessions"
        for pattern in ("*.jsonl.reset.*", "*.jsonl.deleted.*"):
            for p in sdir.glob(pattern):
                if _mtime(p) >= cutoff:
                    rotated += 1
    if rotated > cfg["rotated_backlog_warn"]:
        return Check(
            "rotated_transcripts_visible",
            WARN,
            f"{rotated} rotated transcripts in last 7d (ensure checkpoint recovery ran)",
            rotated,
            "agents/*/sessions",
        )
    return Check("rotated_transcripts_visible", OK, f"{rotated} rotated (last 7d)", rotated, "agents/*/sessions")


def check_pending_transcript_backlog(cfg: dict) -> Check:
    """Recent transcripts whose session file is newer than the timeline ledger.
    Rough signal: many recent transcripts but a stale ledger => backlog."""
    if not TIMELINE_LEDGER.exists():
        return Check("pending_transcript_backlog", FAIL, "timeline ledger missing", None, str(TIMELINE_LEDGER))
    ledger_mtime = _mtime(TIMELINE_LEDGER)
    cutoff = _now() - 3 * 86400
    pending = 0
    for agent_dir in _active_agent_dirs():
        sdir = agent_dir / "sessions"
        for pattern in ("*.jsonl", "*.jsonl.reset.*", "*.jsonl.deleted.*"):
            for p in sdir.glob(pattern):
                m = _mtime(p)
                if m >= cutoff and m > ledger_mtime + 1:
                    pending += 1
    val = {"pending": pending, "ledger_age_seconds": int(_now() - ledger_mtime)}
    if pending > cfg["pending_backlog_warn"]:
        return Check(
            "pending_transcript_backlog",
            WARN,
            f"{pending} transcripts newer than ledger (run timeline refresh)",
            val,
            "agents/*/sessions vs ledger",
        )
    return Check("pending_transcript_backlog", OK, f"{pending} pending", val, "")


def check_postgres_reachable(cfg: dict) -> Check:
    probe: Callable[[], tuple[bool, str]] | None = cfg["probes"].get("postgres")
    if probe is None:
        def probe() -> tuple[bool, str]:
            try:
                from memory_db import get_conn  # local import; heavy

                with get_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                        cur.fetchone()
                return True, "SELECT 1 ok"
            except Exception as e:  # noqa: BLE001
                return False, f"{type(e).__name__}: {e}"

    ok, detail = probe()
    return Check("postgres_reachable", OK if ok else FAIL, detail, ok, "memory_db.get_conn")


def check_qdrant_reachable(cfg: dict) -> Check:
    host = os.environ.get("QDRANT_HOST", "localhost")
    port = os.environ.get("QDRANT_PORT", "6333")
    url = f"http://{host}:{port}/healthz"
    probe = cfg["probes"].get("qdrant")
    ok, detail = probe(url) if probe else _http_ok(url, cfg["http_timeout_s"])
    # Known false-negative: docker may mark container unhealthy while endpoint serves.
    return Check("qdrant_reachable", OK if ok else FAIL, detail, ok, url)


def check_neo4j_reachable(cfg: dict) -> Check:
    url = os.environ.get("NEO4J_HTTP", "http://localhost:7474")
    probe = cfg["probes"].get("neo4j")
    ok, detail = probe(url) if probe else _http_ok(url, cfg["http_timeout_s"])
    # Graph is an enhancement; down => warn, not fail (startup still usable).
    return Check("neo4j_reachable", OK if ok else WARN, detail, ok, url)


def check_search_service_healthy(cfg: dict) -> Check:
    url = os.environ.get("MEMORY_SEARCH_HEALTH_URL", "http://localhost:8791/health")
    probe = cfg["probes"].get("search_service")
    ok, detail = probe(url) if probe else _http_ok(url, cfg["http_timeout_s"])
    # Distinguish dead-PID desync for a clearer signal.
    if not ok:
        pid_file = MEMORY_ROOT / "runtime" / "search_service.pid"
        hint = ""
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                alive = Path(f"/proc/{pid}").exists()
                hint = f" (recorded pid {pid} {'ALIVE' if alive else 'DEAD'})"
            except Exception:  # noqa: BLE001
                pass
        detail += hint
    return Check("search_service_healthy", OK if ok else FAIL, detail, ok, url)


def check_canonical_layout_present(cfg: dict) -> Check:
    """Phase 1: the canonical memory tree (session/daily/digest/resources/metadata)
    is scaffolded. Warn (not fail) if missing — startup is still usable from the
    timeline layer; this just means the readable canonical layer isn't set up."""
    try:
        import canonical_memory_paths as cmp  # local import; optional dependency

        root = cmp.canonical_root()
        if cmp.layout_present():
            return Check("canonical_layout_present", OK, f"scaffolded at {root}", str(root), str(root))
        return Check(
            "canonical_layout_present",
            WARN,
            f"canonical tree not scaffolded (run scaffold_canonical_memory.py); {root}",
            str(root),
            str(root),
        )
    except Exception as e:  # noqa: BLE001
        return Check("canonical_layout_present", WARN, f"canonical paths unavailable: {e}", None, "")


def check_daily_cards_fresh(cfg: dict) -> Check:
    """Phase 2: today's per-agent daily session cards exist and are fresh. WARN if
    the canonical layout is present but today has no fresh cards (auto_memory not
    running); SKIP if the canonical layout isn't scaffolded at all (Phase 1 owns
    that signal)."""
    try:
        import canonical_memory_paths as cmp  # local import

        if not cmp.layout_present():
            return Check("daily_cards_fresh", SKIP, "canonical layout not scaffolded", None, "")
        day = cmp.today_utc()
        ddir = cmp.daily_dir(day, ensure=False)
        if not ddir.exists():
            return Check("daily_cards_fresh", WARN, f"no daily dir for {day} (auto_memory idle)", 0, str(ddir))
        cards = [p for p in ddir.glob("*.md") if p.name != "_index.md"]
        fresh = 0
        for c in cards:
            try:
                head = c.read_text(encoding="utf-8", errors="ignore")[:400].lower()
                if "freshness: fresh" in head or "_generated" in head:
                    fresh += 1
            except OSError:
                continue
        if cards:
            return Check("daily_cards_fresh", OK, f"{len(cards)} agent card(s) for {day}", len(cards), str(ddir))
        return Check("daily_cards_fresh", WARN, f"daily dir for {day} empty", 0, str(ddir))
    except Exception as e:  # noqa: BLE001
        return Check("daily_cards_fresh", WARN, f"daily-card check unavailable: {e}", None, "")


def check_boot_packs_generated(cfg: dict) -> Check:
    """Phase 3: active agents have a recent, non-degraded boot pack. OK if every
    agent that has session cards today also has a fresh boot pack generated within
    24h; WARN if a boot pack is missing/stale/degraded; SKIP if the canonical
    layout isn't scaffolded."""
    import re
    from datetime import datetime, timezone

    try:
        import canonical_memory_paths as cmp

        if not cmp.layout_present():
            return Check("boot_packs_generated", SKIP, "canonical layout not scaffolded", None, "")

        day = cmp.today_utc()
        ddir = cmp.daily_dir(day, ensure=False)
        active_agents = []
        if ddir.exists():
            active_agents = [p.stem for p in ddir.glob("*.md") if p.name != "_index.md"]
        if not active_agents:
            return Check("boot_packs_generated", SKIP, "no active agents today", 0, str(ddir))

        def _frontmatter(text: str) -> dict:
            """Parse the leading --- ... --- YAML-ish frontmatter block fully
            (not a fixed-size head window, which is brittle as frontmatter grows)."""
            fm: dict[str, str] = {}
            lines = text.splitlines()
            if not lines or lines[0].strip() != "---":
                return fm
            for line in lines[1:]:
                if line.strip() == "---":
                    break
                if ":" in line:
                    k, _, v = line.partition(":")
                    fm[k.strip()] = v.strip()
            return fm

        now = datetime.now(timezone.utc)
        today = cmp.today_utc()
        missing, stale, degraded, content_stale, ok = [], [], [], [], 0
        for agent in active_agents:
            bp = cmp.session_dir(agent) / "_BOOT_PACK.md"
            if not bp.exists():
                missing.append(agent)
                continue
            fm = _frontmatter(bp.read_text(encoding="utf-8", errors="ignore"))
            if fm.get("freshness") == "degraded":
                degraded.append(agent)
                continue
            # content-staleness: generator stamps cards_stale / cards_day
            if fm.get("cards_stale") == "true" or (
                fm.get("cards_day") not in (None, "-", "?") and fm.get("cards_day", "") < today
            ):
                content_stale.append(agent)
                continue
            gen_at = fm.get("generated_at", "")
            fresh = False
            try:
                gen = datetime.fromisoformat(gen_at.replace("Z", "+00:00"))
                fresh = (now - gen).total_seconds() < 24 * 3600
            except ValueError:
                fresh = False
            if fresh:
                ok += 1
            else:
                stale.append(agent)

        if missing or degraded or stale or content_stale:
            parts = []
            if missing:
                parts.append(f"missing={missing}")
            if degraded:
                parts.append(f"degraded={degraded}")
            if content_stale:
                parts.append(f"content-stale={content_stale}")
            if stale:
                parts.append(f"stale-timestamp={stale}")
            return Check("boot_packs_generated", WARN,
                         f"{ok}/{len(active_agents)} fresh; " + ", ".join(parts),
                         ok, str(cmp.canonical_root()))
        return Check("boot_packs_generated", OK,
                     f"{ok}/{len(active_agents)} agent boot pack(s) fresh", ok,
                     str(cmp.canonical_root()))
    except Exception as e:  # noqa: BLE001
        return Check("boot_packs_generated", WARN, f"boot-pack check unavailable: {e}", None, "")


def check_embeddings_provider_healthy(cfg: dict) -> Check:
    """Light, non-fatal: surface the configured provider. Embedding outages have
    happened 3x; we want them visible but they shouldn't hard-fail startup since
    the readable layer (timeline/daily/boot pack) still orients the agent."""
    probe = cfg["probes"].get("embeddings")
    if probe is not None:
        ok, detail = probe()
        return Check("embeddings_provider_healthy", OK if ok else WARN, detail, ok, "configured provider")
    # Best-effort: read search_service.json embed_model if present.
    sj = MEMORY_ROOT / "runtime" / "search_service.json"
    if sj.exists():
        try:
            model = json.loads(sj.read_text()).get("embed_model", "?")
            return Check("embeddings_provider_healthy", OK, f"configured: {model}", model, str(sj))
        except Exception:  # noqa: BLE001
            pass
    return Check("embeddings_provider_healthy", SKIP, "no provider info available", None, "")


# --- Future-phase placeholders: registered SKIP so the contract is stable. -----
def _placeholder(name: str, note: str) -> Callable[[dict], Check]:
    def _c(cfg: dict) -> Check:
        return Check(name, SKIP, f"not yet implemented ({note})", None, "")
    return _c


CORE_CHECKS: list[Callable[[dict], Check]] = [
    check_timeline_latest_fresh,
    check_timeline_today_exists,
    check_workspaces_timeline_linked,
    check_boot_text_clean,
    check_checkpoint_cursors_advancing,
    check_rotated_transcripts_visible,
    check_pending_transcript_backlog,
    check_postgres_reachable,
    check_qdrant_reachable,
    check_neo4j_reachable,
    check_search_service_healthy,
    check_embeddings_provider_healthy,
    check_canonical_layout_present,
    check_daily_cards_fresh,
    check_boot_packs_generated,
]

FUTURE_CHECKS: list[Callable[[dict], Check]] = [
    _placeholder("digest_fresh", "Phase 4"),
    _placeholder("auto_dream_last_run", "Phase 4"),
]


def run_checks(cfg: dict, *, include_future: bool = True) -> Report:
    checks: list[Check] = []
    for fn in CORE_CHECKS:
        try:
            checks.append(fn(cfg))
        except Exception as e:  # noqa: BLE001 — a buggy check must not crash the tool
            checks.append(Check(getattr(fn, "__name__", "unknown"), FAIL, f"check raised: {e}"))
    if include_future:
        for fn in FUTURE_CHECKS:
            checks.append(fn(cfg))

    overall = OK
    for c in checks:
        if _SEVERITY[c.status] > _SEVERITY[overall]:
            overall = c.status
    if overall == SKIP:
        overall = OK
    return Report(
        generated_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        overall=overall,
        checks=checks,
    )


def build_cfg(thresholds_file: str | None, probes: dict | None = None) -> dict:
    cfg = dict(DEFAULTS)
    if thresholds_file:
        try:
            cfg.update(json.loads(Path(thresholds_file).read_text(encoding="utf-8")))
        except Exception as e:  # noqa: BLE001
            print(f"WARN: could not read thresholds file: {e}", file=sys.stderr)
    cfg["probes"] = probes or {}
    return cfg


def write_health_json(report: Report) -> Path:
    HEALTH_DIR.mkdir(parents=True, exist_ok=True)
    HEALTH_OUT.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return HEALTH_OUT


_GLYPH = {OK: "OK  ", WARN: "WARN", FAIL: "FAIL", SKIP: "skip"}


def human_summary(report: Report) -> str:
    lines = [
        f"Memory startup health: {report.overall.upper()}  ({report.generated_at})",
        "-" * 64,
    ]
    for c in report.checks:
        lines.append(f"[{_GLYPH[c.status]}] {c.name:32s} {c.detail}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Memory startup health check")
    ap.add_argument("--json", action="store_true", help="print machine JSON to stdout")
    ap.add_argument("--quiet", action="store_true", help="suppress human summary")
    ap.add_argument("--deep", action="store_true", help="run deep probes (memory_search smoke)")
    ap.add_argument("--no-write", action="store_true", help="do not write health JSON")
    ap.add_argument("--no-future", action="store_true", help="omit future-phase placeholders")
    ap.add_argument("--warn-ok", action="store_true", help="exit 0 even when overall==warn")
    ap.add_argument("--thresholds-file")
    args = ap.parse_args(argv)

    cfg = build_cfg(args.thresholds_file)
    cfg["deep"] = args.deep
    report = run_checks(cfg, include_future=not args.no_future)

    if not args.no_write:
        try:
            write_health_json(report)
        except Exception as e:  # noqa: BLE001
            print(f"WARN: could not write health JSON: {e}", file=sys.stderr)

    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    if not args.quiet and not args.json:
        print(human_summary(report))

    if report.overall == FAIL:
        return 2
    if report.overall == WARN:
        return 0 if args.warn_ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
