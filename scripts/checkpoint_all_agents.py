"""
Checkpoint all agents state for crash recovery.

Key functions: find_latest_session_for_agent, list_agents_with_sessions, main
"""
import json
import subprocess
import hashlib
import argparse
import time
from pathlib import Path

WORKSPACE = Path.home() / ".openclaw" / "workspace"
OPENCLAW_ROOT = Path.home() / ".openclaw"
AGENTS_DIR = OPENCLAW_ROOT / "agents"
SCRIPTS_DIR = WORKSPACE / ".memory-index" / "scripts"
LOGS_DIR = WORKSPACE / ".memory-index" / "logs"
CHECKPOINT_SCRIPT = SCRIPTS_DIR / "checkpoint_agent.py"
STATE_FILE = LOGS_DIR / "checkpoint_all_agents_state.json"

LOGS_DIR.mkdir(parents=True, exist_ok=True)

def discover_session_files_for_agent(agent_name: str):
    """Return all active and rotated/reset transcript candidates for an agent.

    The previous implementation selected only the latest active *.jsonl file,
    which can skip large rotated files named *.jsonl.reset.<timestamp>.
    """
    sessions_dir = AGENTS_DIR / agent_name / "sessions"
    sessions_json = sessions_dir / "sessions.json"
    by_path = {}

    if sessions_json.exists():
        try:
            data = json.loads(sessions_json.read_text(encoding="utf-8"))
            for _, meta in data.items():
                session_file = meta.get("sessionFile")
                updated_at = meta.get("updatedAt", 0)
                if session_file:
                    p = Path(session_file)
                    if p.exists():
                        by_path[p.resolve()] = (int(updated_at or 0), p)
        except Exception:
            pass

    for pattern in ("*.jsonl", "*.jsonl.reset.*", "*.jsonl.deleted.*"):
        for p in sessions_dir.glob(pattern):
            try:
                mtime = int(p.stat().st_mtime * 1000)
            except Exception:
                mtime = 0
            by_path[p.resolve()] = (mtime, p)

    rows = list(by_path.values())
    rows.sort(key=lambda x: x[0])  # oldest first so backlog drains predictably
    return [p for _, p in rows]


def list_agents_with_sessions(since_ms: int | None = None):
    rows = []
    if not AGENTS_DIR.exists():
        return rows

    for agent_dir in sorted(AGENTS_DIR.iterdir()):
        if not agent_dir.is_dir():
            continue
        agent_name = agent_dir.name
        for path in discover_session_files_for_agent(agent_name):
            if since_ms is not None:
                try:
                    if int(path.stat().st_mtime * 1000) < since_ms:
                        continue
                except Exception:
                    continue
            rows.append((agent_name, path))
    return rows


def load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def session_signature(path: Path) -> dict:
    st = path.stat()
    return {
        "session_file": str(path),
        "mtime_ms": int(st.st_mtime * 1000),
        "size": int(st.st_size),
    }


def state_key(agent_name: str, path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:16]
    return f"{agent_name}:{path.name}:{digest}"


def should_checkpoint(agent_name: str, latest: Path, state: dict) -> bool:
    sig = session_signature(latest)
    prev = state.get(state_key(agent_name, latest)) or {}
    return not (
        prev.get("session_file") == sig["session_file"]
        and prev.get("mtime_ms") == sig["mtime_ms"]
        and prev.get("size") == sig["size"]
    )


def processing_priority(row: tuple[str, Path]) -> tuple[int, int, str, str]:
    """Prioritize likely-loss files when a backlog cap is active.

    Reset/rotated transcripts are the data-loss class that prompted this fix,
    so process them before ordinary boot/active transcripts. Within each class,
    process newest first to capture recent user-visible gaps quickly.
    """
    agent_name, path = row
    is_reset = ".jsonl.reset." in path.name or ".jsonl.deleted." in path.name
    try:
        mtime_ms = int(path.stat().st_mtime * 1000)
    except Exception:
        mtime_ms = 0
    return (0 if is_reset else 1, -mtime_ms, agent_name, path.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="List pending transcript checkpoints without processing")
    parser.add_argument("--max-files", type=int, default=25, help="Maximum changed transcript files to process in one run (default: 25)")
    parser.add_argument("--since-days", type=float, default=7.0, help="Only consider transcripts modified in the last N days; use 0 for all (default: 7)")
    parser.add_argument(
        "--max-total-seconds", type=int, default=240,
        help="Stop selecting new files after this many seconds of total runtime (default: 240 = 4 min). "
             "Ensures the timer can re-fire promptly instead of blocking for hours on backlog.",
    )
    parser.add_argument(
        "--per-file-timeout", type=int, default=120,
        help="Kill a single checkpoint_agent.py subprocess after this many seconds (default: 120). "
             "Prevents one giant transcript from starving the rest of the queue.",
    )
    args = parser.parse_args()

    run_start = time.monotonic()

    since_ms = None
    if args.since_days and args.since_days > 0:
        since_ms = int((time.time() - args.since_days * 86400) * 1000)

    rows = list_agents_with_sessions(since_ms=since_ms)
    if not rows:
        print("No agent sessions found.")
        return

    state = load_state()

    pending_rows = [(agent_name, path) for agent_name, path in rows if should_checkpoint(agent_name, path, state)]
    pending_rows.sort(key=processing_priority)
    selected_pending = pending_rows[:max(0, args.max_files)] if args.max_files > 0 else pending_rows

    print("Agent session transcripts discovered:")
    print(f"- considered={len(rows)} pending={len(pending_rows)} selected={len(selected_pending)}"
          f" since_days={args.since_days} max_files={args.max_files}"
          f" max_total_seconds={args.max_total_seconds} per_file_timeout={args.per_file_timeout}")
    for agent_name, path in rows:
        marker = "checkpoint" if (agent_name, path) in selected_pending else ("pending_later" if (agent_name, path) in pending_rows else "skip_unchanged")
        print(f"- {agent_name}: {path} [{marker}]")
    print()

    if args.dry_run:
        print("DRY_RUN: no checkpoint processing performed.")
        return

    changed = 0
    skipped = 0
    timed_out_files = 0
    budget_exhausted = False

    for agent_name, latest in rows:
        if (agent_name, latest) not in selected_pending:
            print(f"===== SKIP {agent_name} =====")
            print("No session changes since last checkpoint or deferred by per-run limit.")
            print()
            skipped += 1
            continue

        # ── Runtime budget check ──────────────────────────────────────
        elapsed = time.monotonic() - run_start
        if elapsed >= args.max_total_seconds:
            remaining_pending = sum(
                1 for a, p in selected_pending
                if should_checkpoint(a, p, state)
            )
            print(f"===== BUDGET EXHAUSTED after {elapsed:.0f}s ====="
                  f" (limit={args.max_total_seconds}s,"
                  f" remaining_pending={remaining_pending})")
            budget_exhausted = True
            break

        print(f"===== CHECKPOINT {agent_name} ===== [{elapsed:.0f}s elapsed]")

        try:
            result = subprocess.run(
                [
                    "python3",
                    str(CHECKPOINT_SCRIPT),
                    "--agent", agent_name,
                    "--transcript", str(latest),
                    "--reason", f"Cross-agent checkpoint ({agent_name}: {latest.name})"
                ],
                text=True,
                capture_output=True,
                timeout=args.per_file_timeout,
            )
        except subprocess.TimeoutExpired:
            print(f"TIMEOUT after {args.per_file_timeout}s — skipping {latest.name}")
            print("This file will be retried on the next cycle.")
            timed_out_files += 1
            print()
            continue

        if result.stdout:
            print(result.stdout.strip())
        if result.stderr:
            print("STDERR:")
            print(result.stderr.strip())

        if result.returncode == 0:
            state[state_key(agent_name, latest)] = session_signature(latest)
            changed += 1
        else:
            print(f"Checkpoint failed for {agent_name}; state not advanced.")

        print()

    save_state(state)
    total_elapsed = time.monotonic() - run_start
    print(f"Summary: checkpointed={changed} skipped_unchanged={skipped}"
          f" timed_out={timed_out_files} budget_exhausted={budget_exhausted}"
          f" elapsed={total_elapsed:.0f}s total={len(rows)}")

if __name__ == "__main__":
    main()
