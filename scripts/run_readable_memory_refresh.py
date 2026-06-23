#!/usr/bin/env python3
"""
Memory system utility: refresh the entire READABLE memory layer in one command.

Single orchestrator for Phases 2–6. Chains the five existing generators in
dependency order, isolates per-step failures, logs structured output, and applies
a HEALTH GATE that can actually detect a readable-layer regression (the whole
point). Intended to be the one command a runtime boot hook / cron / human calls.

Pipeline (order is dependency-correct, must not reorder):
  1. generate_daily_session_card.py --all-agents   (session transcript -> daily card)
  2. consolidate_memory_digest.py   --all-agents   (daily -> digest)
  3. generate_agent_boot_pack.py    --all-agents   (reads daily cards + timeline;
                                                    does NOT read digests)
  4. card_search_index.py                          (indexes session+daily+digest)
  5. generate_memory_dashboard.py                  (reads everything + health)
  6. memory_health_check.py --no-write             (post-condition GATE)

THE GATE (why it's built this way): all five readable-layer health checks only ever
emit WARN/SKIP on regression (never FAIL), and the live tree often sits at
overall=WARN from an unrelated rotated-transcripts check. So judging on health
`overall` is useless — it can't tell "readable layer regressed" from "unrelated
WARN". Instead the gate evaluates the FIVE NAMED checks individually, combined with
whether the step that builds each artifact ran: "we just (re)built X and the check
still says X is missing/empty" => FAIL; "X present but stale" => DEGRADED; SKIP
(empty/unscaffolded tree) => PASS.

Exit codes: 0 = ok, 1 = failed, 2 = degraded (and "locked"). degraded is a distinct
nonzero so a cron/boot wrapper can treat rc in (0,2) as non-fatal yet still ALERT
on 2, without parsing JSON.

This orchestrator does NOT wire itself into boot/cron (that's the runtime's
decision) and does NOT reimplement agent discovery (delegates to --all-agents).

Key functions: run_refresh (testable core), evaluate_gate, acquire_lock.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

LOGS_DIR = SCRIPT_DIR.parent / "logs"
LOCKFILE = LOGS_DIR / ".readable-refresh.lock"
DEFAULT_STEP_TIMEOUT = 300
DEFAULT_LOCK_STALE_MIN = 30

OK, DEGRADED, FAILED, LOCKED = "ok", "degraded", "failed", "locked"
EXIT = {OK: 0, FAILED: 1, DEGRADED: 2, LOCKED: 2}

# The five readable-layer checks and which pipeline step builds each artifact.
# step index is into the STEPS list below (0-based). Used by the gate to decide
# "did the builder for this artifact run?".
GATE_CHECKS = {
    "daily_cards_fresh": 0,
    "digest_fresh": 1,
    "boot_packs_generated": 2,
    "card_search_ready": 3,
    "dashboard_fresh": 4,
}
# Tokens in a check's detail that mean the artifact is MISSING/EMPTY (vs merely
# stale). Kept in ONE place: a future check-wording change must be caught by the
# token test rather than silently misclassifying a stale artifact as a failure.
# NOTE (digest): missing and stale BOTH report value==0, and the consolidate step
# JSON only exposes nodes_written (new-this-run) — so for digest the missing-vs-
# stale split rests ENTIRELY on these text tokens ("no ... yet" vs "none fresh").
_MISSING_TOKENS = ("no digest nodes yet", "no agent card", "no boot pack",
                   "missing", "card index missing", "dashboard missing",
                   "index empty", "0 records", "(0 ", "none yet")
# Tokens that mean STALE (artifact present, just old). Used to make the value==0
# fallback PRECISE: a value==0 only counts as "missing" when the detail is NOT a
# known staleness phrase. This avoids the broad `"fresh" not in detail` substring,
# which would also match unrelated words like "refreshed". The only real value==0
# stale case today is digest "none fresh"; the rest carry an explicit count>0.
_STALE_TOKENS = ("none fresh", "stale", "generated ", "newest ", "no fresh")


@dataclass
class StepResult:
    index: int
    name: str
    cmd: list[str]
    returncode: int | None
    ok: bool
    duration_s: float
    reason: str = ""
    stdout_tail: str = ""
    stderr_tail: str = ""
    parsed_json: dict | None = None


# ---------------------------------------------------------------------------
# Step definitions (script, args). --all-agents on per-agent steps.
# ---------------------------------------------------------------------------
def build_steps(since_days: float) -> list[tuple[str, list[str]]]:
    sd = str(since_days)
    return [
        ("generate_daily_session_card.py", ["--all-agents", "--since-days", sd, "--json"]),
        ("consolidate_memory_digest.py", ["--all-agents", "--since-days", sd, "--json"]),
        ("generate_agent_boot_pack.py", ["--all-agents", "--json"]),
        ("card_search_index.py", ["--json"]),
        ("generate_memory_dashboard.py", ["--json"]),
    ]


def _default_runner(cmd: list[str], timeout: int) -> tuple[int | None, str, str, str]:
    """Execute a step. Returns (returncode, stdout, stderr, reason).

    returncode is None on timeout/spawn failure (reason set). Isolated here so tests
    can inject a fake runner and avoid spawning the heavy generators."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr, ""
    except subprocess.TimeoutExpired as e:
        return None, e.stdout or "", e.stderr or "", "timeout"
    except OSError as e:
        return None, "", str(e), f"spawn-error: {e}"


def _tail(s: str, n: int = 2000) -> str:
    s = s or ""
    return s if len(s) <= n else s[-n:]


# ---------------------------------------------------------------------------
# Lockfile (advisory, with stale detection)
# ---------------------------------------------------------------------------
def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but not ours
    except OSError:
        return False
    return True


def _lock_payload() -> str:
    return json.dumps({"pid": os.getpid(),
                       "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})


def _try_exclusive_create(lock_path: Path) -> bool:
    """Atomically create the lockfile iff it does not exist (O_CREAT|O_EXCL).

    This is the race-free fast path: two concurrent racers can't both win the
    create, eliminating the check-then-write TOCTOU on the common (no prior lock)
    path. Returns True if WE created it, False if it already existed."""
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    try:
        os.write(fd, _lock_payload().encode("utf-8"))
    finally:
        os.close(fd)
    return True


def acquire_lock(lock_path: Path, *, stale_minutes: int, force: bool) -> tuple[bool, str]:
    """Try to acquire the advisory lock. Returns (acquired, note).

    Fast path is an atomic exclusive create (race-free). If the lock already exists,
    steal it only if --force, the holder PID is dead, or it's older than
    stale_minutes (guards against a crash wedging all future runs). The steal path
    remains a non-atomic advisory best-effort, which is acceptable per the plan."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if _try_exclusive_create(lock_path):
        return True, ""

    # Lock already present: decide whether to steal it.
    note = ""
    steal = force
    if force:
        note = "forced lock steal"
    try:
        info = json.loads(lock_path.read_text(encoding="utf-8"))
        holder_pid = int(info.get("pid", -1))
    except Exception:  # noqa: BLE001
        holder_pid = -1
    try:
        age_min = (time.time() - lock_path.stat().st_mtime) / 60.0
    except OSError:
        # vanished between create-attempt and stat: retry the atomic create once
        return (True, "") if _try_exclusive_create(lock_path) else \
            (False, "lock contended")
    if not steal and holder_pid > 0 and not _pid_alive(holder_pid):
        steal, note = True, f"stale lock stolen: holder pid {holder_pid} dead"
    if not steal and age_min > stale_minutes:
        steal, note = True, f"stale lock stolen: age {age_min:.1f}m > {stale_minutes}m"
    if not steal:
        return False, f"lock held by pid {holder_pid} (age {age_min:.1f}m)"
    _write_lock(lock_path)  # overwrite with our ownership
    return True, note


def _write_lock(lock_path: Path) -> None:
    lock_path.write_text(_lock_payload(), encoding="utf-8")


def release_lock(lock_path: Path) -> None:
    try:
        if lock_path.exists():
            info = json.loads(lock_path.read_text(encoding="utf-8"))
            if int(info.get("pid", -1)) == os.getpid():
                lock_path.unlink()
    except Exception:  # noqa: BLE001
        pass  # best-effort release; stale detection covers a leftover


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------
def _is_missing(detail: str, value) -> bool:
    d = (detail or "").lower()
    if any(tok in d for tok in _MISSING_TOKENS):
        return True
    # value==0 means empty for the count-based checks; but for digest_fresh value==0
    # is shared by stale ("none fresh"), so only treat value==0 as missing when the
    # detail does NOT match a known staleness phrase (precise, not a broad "fresh"
    # substring that would also catch "refreshed").
    if value == 0 and not any(tok in d for tok in _STALE_TOKENS):
        return True
    return False


def evaluate_gate(health: dict, steps: list[StepResult]) -> dict:
    """Per-artifact verdict from the five named checks + whether their step ran.

    Returns {per_artifact:{name:{check_status, step_ok, verdict}}, summary}.
    verdict ∈ {pass, degraded, fail}. Rollup: any fail -> fail; any degraded ->
    degraded; else pass. Health that couldn't be evaluated yields 'unknown' summary.
    """
    by_name = {c.get("name"): c for c in health.get("checks", [])}
    per: dict[str, dict] = {}
    for cname, step_idx in GATE_CHECKS.items():
        c = by_name.get(cname)
        step_ran = step_idx < len(steps) and steps[step_idx] is not None
        if c is None:
            per[cname] = {"check_status": "absent", "step_ok": step_ran, "verdict": "degraded"}
            continue
        status = c.get("status")
        detail, value = c.get("detail", ""), c.get("value")
        if status == "skip":
            verdict = "pass"  # nothing to build (unscaffolded/empty tree)
        elif status == "ok":
            verdict = "pass"
        elif status == "fail":
            verdict = "fail"
        elif status == "warn":
            if _is_missing(detail, value) and step_ran:
                verdict = "fail"   # we just built it and it's still missing/empty
            else:
                verdict = "degraded"  # staleness / unrelated
        else:
            verdict = "degraded"
        per[cname] = {"check_status": status, "step_ok": step_ran, "verdict": verdict}

    verdicts = [v["verdict"] for v in per.values()]
    if "fail" in verdicts:
        summary = "fail"
    elif "degraded" in verdicts:
        summary = "degraded"
    else:
        summary = "pass"
    return {"per_artifact": per, "summary": summary}


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
def run_refresh(
    *,
    since_days: float = 1.0,
    continue_on_error: bool = True,
    skip_health: bool = False,
    log_dir: Path | None = None,
    step_timeout: int = DEFAULT_STEP_TIMEOUT,
    runner=None,
    health_probe=None,
    lock: bool = True,
    force: bool = False,
    lock_stale_minutes: int = DEFAULT_LOCK_STALE_MIN,
) -> dict:
    """Run the readable-layer refresh. Returns a summary dict (with 'overall').

    runner(cmd, timeout)->(rc,out,err,reason) and health_probe()->health_dict are
    injectable for hermetic testing. lock=False disables the lockfile (tests).
    """
    runner = runner or _default_runner
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # log dir — fail FAST if uncreatable, before any step
    ts = started.replace(":", "-")
    ld = log_dir or (LOGS_DIR / f"readable-refresh-{ts}")
    try:
        ld.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return {"generated_at": started, "overall": FAILED, "exit_code": EXIT[FAILED],
                "error": f"log-dir uncreatable: {e}", "steps": [], "gate": None}

    # lock
    lock_note = ""
    if lock:
        acquired, lock_note = acquire_lock(LOCKFILE, stale_minutes=lock_stale_minutes, force=force)
        if not acquired:
            return {"generated_at": started, "overall": LOCKED, "exit_code": EXIT[LOCKED],
                    "note": lock_note, "steps": [], "gate": None}

    try:
        steps_def = build_steps(since_days)
        results: list[StepResult] = []
        aborted = False
        for i, (script, args) in enumerate(steps_def):
            if aborted:
                break
            cmd = [sys.executable, str(SCRIPT_DIR / script), *args]
            t0 = time.time()
            rc, out, err, reason = runner(cmd, step_timeout)
            dur = round(time.time() - t0, 3)
            ok = rc == 0
            parsed = None
            if out:
                try:
                    parsed = json.loads(out.strip().splitlines()[-1]) if out.strip() else None
                    if not isinstance(parsed, dict):
                        parsed = None
                except (json.JSONDecodeError, IndexError):
                    parsed = None
            sr = StepResult(index=i, name=script, cmd=cmd, returncode=rc, ok=ok,
                            duration_s=dur, reason=reason, stdout_tail=_tail(out),
                            stderr_tail=_tail(err), parsed_json=parsed)
            results.append(sr)
            # per-step logs
            (ld / f"{i:02d}-{script}.out").write_text(out or "", encoding="utf-8")
            (ld / f"{i:02d}-{script}.err").write_text(err or "", encoding="utf-8")
            if not ok and not continue_on_error:
                aborted = True

        # health gate
        gate = None
        health = {}
        if not skip_health and not aborted:
            try:
                if health_probe is not None:
                    health = health_probe()
                else:
                    hcmd = [sys.executable, str(SCRIPT_DIR / "memory_health_check.py"),
                            "--json", "--no-write"]
                    hp = subprocess.run(hcmd, capture_output=True, text=True, timeout=step_timeout)
                    health = json.loads(hp.stdout) if hp.stdout.strip() else {}
                gate = evaluate_gate(health, results)
            except Exception as e:  # noqa: BLE001
                gate = {"per_artifact": {}, "summary": "unknown", "error": str(e)}

        overall = _rollup(results, gate, aborted=aborted, skip_health=skip_health)
        summary = {
            "generated_at": started,
            "since_days": since_days,
            "overall": overall,
            "exit_code": EXIT[overall],
            "aborted": aborted,
            "lock_note": lock_note,
            "steps": [_step_public(s) for s in results],
            "gate": gate,
            "health_overall": health.get("overall"),
            "log_dir": str(ld),
        }
        (ld / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                         encoding="utf-8")
        return summary
    finally:
        if lock:
            release_lock(LOCKFILE)


def _step_public(s: StepResult) -> dict:
    d = asdict(s)
    d.pop("cmd", None)  # cmd has absolute paths; keep logs lean
    return d


def _rollup(results: list[StepResult], gate: dict | None, *, aborted: bool, skip_health: bool) -> str:
    # hard-stop abort under --no-continue-on-error
    if aborted:
        return FAILED
    gate_summary = (gate or {}).get("summary")
    any_step_failed = any(not s.ok for s in results)
    if gate_summary == "fail":
        return FAILED
    if gate_summary == "pass":
        # a step failed but artifacts are fine -> degraded, else ok
        return DEGRADED if any_step_failed else OK
    if gate_summary == "degraded":
        return DEGRADED
    # gate unknown / skipped: judge by steps + step JSON only
    if skip_health:
        return DEGRADED if any_step_failed else OK
    # health couldn't be evaluated and we didn't skip it: don't claim success blindly
    return DEGRADED if not any_step_failed else FAILED


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Refresh the readable memory layer (Phases 2–6).")
    ap.add_argument("--since-days", type=float, default=1.0)
    ap.add_argument("--skip-health", action="store_true",
                    help="skip the health gate (FORFEITS regression detection)")
    ap.add_argument("--no-continue-on-error", dest="continue_on_error",
                    action="store_false", help="stop at first failing step")
    ap.add_argument("--force", action="store_true", help="steal a held/stale lockfile")
    ap.add_argument("--lock-stale-minutes", type=int, default=DEFAULT_LOCK_STALE_MIN)
    ap.add_argument("--log-dir")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    summary = run_refresh(
        since_days=args.since_days,
        continue_on_error=args.continue_on_error,
        skip_health=args.skip_health,
        log_dir=Path(args.log_dir) if args.log_dir else None,
        force=args.force,
        lock_stale_minutes=args.lock_stale_minutes,
    )

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    elif not args.quiet:
        print(f"readable refresh: {summary['overall'].upper()} "
              f"(exit {summary['exit_code']}) -> {summary.get('log_dir', '?')}")
        if summary.get("note"):
            print(f"  {summary['note']}")
        for s in summary.get("steps", []):
            flag = "ok" if s["ok"] else f"FAIL({s.get('reason') or s.get('returncode')})"
            print(f"  [{flag}] {s['name']}")
        g = summary.get("gate")
        if g and g.get("per_artifact"):
            for name, v in g["per_artifact"].items():
                print(f"  gate {v['verdict']:8} {name} ({v['check_status']})")
    return summary["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
