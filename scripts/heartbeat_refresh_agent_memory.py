"""
Heartbeat subsystem: refresh agent memory management.

Key functions: main
"""
import argparse
import subprocess
import time
from pathlib import Path

WORKSPACE = Path.home() / ".openclaw" / "workspace"
OPENCLAW_ROOT = Path.home() / ".openclaw"
AGENTS_DIR = OPENCLAW_ROOT / "agents"
SCRIPTS_DIR = WORKSPACE / ".memory-index" / "scripts"
REFRESH_SCRIPT = SCRIPTS_DIR / "refresh_agent_memory.py"
TIMELINE_LEDGER_SCRIPT = SCRIPTS_DIR / "timeline_event_ledger.py"
TIMELINE_RENDER_SCRIPT = SCRIPTS_DIR / "render_timeline_views.py"
HEALTH_CHECK_SCRIPT = SCRIPTS_DIR / "memory_health_check.py"


def recent_transcripts(*, since_days: float) -> list[tuple[str, Path]]:
    cutoff = time.time() - since_days * 86400
    rows: list[tuple[str, Path]] = []
    if not AGENTS_DIR.exists():
        return rows

    patterns = ("*.jsonl", "*.jsonl.reset.*", "*.jsonl.deleted.*")
    for agent_dir in sorted(AGENTS_DIR.iterdir()):
        sessions_dir = agent_dir / "sessions"
        if not sessions_dir.is_dir():
            continue
        for pattern in patterns:
            for path in sessions_dir.glob(pattern):
                try:
                    if path.stat().st_mtime < cutoff:
                        continue
                except OSError:
                    continue
                rows.append((agent_dir.name, path))
    rows.sort(key=lambda row: (row[0], str(row[1])))
    return rows


def refresh_timeline(*, since_days: float) -> None:
    """Refresh generated timeline views from recent visible transcript events.

    This is intentionally local/idempotent: timeline_event_ledger.py deduplicates
    by event id, and render_timeline_views.py regenerates daily/latest markdown.
    Include reset/deleted rotations so post-/reset sessions can still answer
    chronology questions like "why did I say PR?" from the real sequence.
    """
    if not TIMELINE_LEDGER_SCRIPT.exists() or not TIMELINE_RENDER_SCRIPT.exists():
        return

    for agent_name, path in recent_transcripts(since_days=since_days):
        try:
            subprocess.run(
                ["python3", str(TIMELINE_LEDGER_SCRIPT), "--agent", agent_name, "--file", str(path)],
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError:
            # Do not block agent memory refresh because one transcript is malformed.
            continue

    subprocess.run(["python3", str(TIMELINE_RENDER_SCRIPT)], capture_output=True, text=True, check=True)


def run_health_check() -> None:
    """Run the memory startup health check and persist its JSON report.

    Intentionally NON-FATAL: a degraded/unhealthy result must surface (written to
    health/memory_startup_latest.json + a WARN line) but must never block agent
    memory refresh. Loudness is the point — silent partial context is the bug we
    are fixing — but we don't trade availability for it here.
    """
    if not HEALTH_CHECK_SCRIPT.exists():
        return
    try:
        # --warn-ok so warn doesn't look like an error here; FAIL still returns 2.
        proc = subprocess.run(
            ["python3", str(HEALTH_CHECK_SCRIPT), "--quiet", "--warn-ok"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 2:
            # Degraded/unsafe startup — make it visible in heartbeat logs.
            print(f"WARN: memory startup health degraded (exit {proc.returncode}); "
                  f"see health/memory_startup_latest.json")
    except Exception as e:  # noqa: BLE001 — health must never break refresh
        print(f"WARN: memory health check failed to run: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-timeline", action="store_true", help="Skip pre-refresh timeline rendering")
    parser.add_argument("--skip-health", action="store_true", help="Skip post-refresh memory health check")
    parser.add_argument("--timeline-since-days", type=float, default=3.0)
    args = parser.parse_args()

    if not args.skip_timeline:
        refresh_timeline(since_days=args.timeline_since_days)

    if not args.skip_health:
        run_health_check()

    cmd = [
        "python3",
        str(REFRESH_SCRIPT),
        "--agent",
        args.agent,
    ]
    if args.force:
        cmd.append("--force")

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print(result.stdout.strip())


if __name__ == "__main__":
    main()
