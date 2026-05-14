"""
Heartbeat subsystem: refresh agent memory management.

Key functions: main
"""
import argparse
import subprocess
from pathlib import Path

WORKSPACE = Path.home() / ".openclaw" / "workspace"
SCRIPTS_DIR = WORKSPACE / ".memory-index" / "scripts"
REFRESH_SCRIPT = SCRIPTS_DIR / "refresh_agent_memory.py"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

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
