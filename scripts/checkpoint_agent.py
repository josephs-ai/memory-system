import json
import argparse
import subprocess
from pathlib import Path

WORKSPACE = Path.home() / ".openclaw" / "workspace"
OPENCLAW_ROOT = Path.home() / ".openclaw"
AGENTS_DIR = OPENCLAW_ROOT / "agents"
SCRIPTS_DIR = WORKSPACE / ".memory-index" / "scripts"
DEHYDRATED_DIR = WORKSPACE / ".memory-index" / "dehydrated"
CHUNKS_DIR = DEHYDRATED_DIR / "chunks_runtime"

DEHYDRATED_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

DEHYDRATE_SCRIPT = SCRIPTS_DIR / "dehydrate_transcript.py"
CHUNK_SCRIPT = SCRIPTS_DIR / "chunk_by_topic.py"
EXTRACT_CHUNK_UPDATES_SCRIPT = SCRIPTS_DIR / "extract_chunk_updates.py"
ROUTE_SCRIPT = SCRIPTS_DIR / "route_memory_items_batch.py"
PROCESS_AUTO_SCRIPT = SCRIPTS_DIR / "process_auto_memory_items.py"

MAINTENANCE_CYCLE_SCRIPT = SCRIPTS_DIR / "run_memory_maintenance_cycle.py"

def find_latest_session_for_agent(agent_name: str):
    sessions_dir = AGENTS_DIR / agent_name / "sessions"
    sessions_json = sessions_dir / "sessions.json"

    candidates = []

    if sessions_json.exists():
        try:
            data = json.loads(sessions_json.read_text(encoding="utf-8"))
            for _, meta in data.items():
                session_file = meta.get("sessionFile")
                updated_at = meta.get("updatedAt", 0)
                if session_file:
                    p = Path(session_file)
                    if p.exists():
                        candidates.append((updated_at, p))
        except Exception:
            pass

    for p in sessions_dir.glob("*.jsonl"):
        try:
            mtime = int(p.stat().st_mtime * 1000)
        except Exception:
            mtime = 0
        candidates.append((mtime, p))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

def run_capture(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout

def clear_old_chunks(chunk_dir: Path):
    for p in chunk_dir.glob("*.txt"):
        p.unlink()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True)
    parser.add_argument("--reason", default="Structured checkpoint run")
    parser.add_argument("--pair-window", type=int, default=3)
    args = parser.parse_args()

    transcript = find_latest_session_for_agent(args.agent)
    if not transcript:
        print(f"No latest session found for agent: {args.agent}")
        return

    dehydrated_file = DEHYDRATED_DIR / f"{args.agent}.latest.txt"
    chunk_dir = CHUNKS_DIR / args.agent
    chunk_dir.mkdir(parents=True, exist_ok=True)
    clear_old_chunks(chunk_dir)

    dehydrated = run_capture([
        "python3",
        str(DEHYDRATE_SCRIPT),
        "--agent",
        args.agent,
    ])
    dehydrated_file.write_text(dehydrated, encoding="utf-8")
    print(f"Saved dehydrated transcript to: {dehydrated_file}")

    run_capture([
        "python3",
        str(CHUNK_SCRIPT),
        "--input",
        str(transcript),
        "--outdir",
        str(chunk_dir),
    ])

    extracted = run_capture([
        "python3",
        str(EXTRACT_CHUNK_UPDATES_SCRIPT),
        "--chunk-dir",
        str(chunk_dir),
        "--source-agent",
        args.agent,
        "--source-session",
        str(transcript.name),
    ]).strip()

    extracted_path = DEHYDRATED_DIR / f"{args.agent}.extracted.txt"
    extracted_path.write_text(extracted + ("\n" if extracted and not extracted.endswith("\n") else ""), encoding="utf-8")

    routed = run_capture([
        "python3",
        str(ROUTE_SCRIPT),
        "--input",
        str(extracted_path),
    ]).strip()

    print("Structured routing output:")
    print(routed)
    print()

    if "No structured items to route." in routed or routed.strip() == "NONE":
        print("Checkpoint policy decision: allow=False reason=no-durable-updates")
        print("Skipping checkpoint due to policy.")
        return

    if "AUTO " in routed or "INBOX " in routed or "DISCARDED " in routed or "SKIP_DUPLICATE " in routed:
        print("Checkpoint policy decision: allow=True reason=structured-items-processed")
        print()

        processed = run_capture([
            "python3",
            str(PROCESS_AUTO_SCRIPT),
        ]).strip()

        print("AUTO processing output:")
        print(processed)
        print()

        maintenance = run_capture([
            "python3",
            str(MAINTENANCE_CYCLE_SCRIPT),
        ]).strip()

        print("Maintenance cycle output:")
        print(maintenance)
        print()
        return
    print("Checkpoint policy decision: allow=False reason=no-actionable-structured-output")
    print("Skipping checkpoint due to policy.")

if __name__ == "__main__":
    main()

