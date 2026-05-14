"""
Memory system utility: write memory.

Key functions: today_file, resolve_target, normalize_text, target_contains_text
"""
import os
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

WORKSPACE = Path(os.environ.get("OPENCLAW_WORKSPACE", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
VENV_PYTHON = WORKSPACE / ".memory-venv" / "bin" / "python"

RECONCILE_SCRIPT = WORKSPACE / ".memory-index" / "scripts" / "reconcile_update.py"

def today_file():
    return WORKSPACE / "memory" / "daily" / f"{datetime.now().date().isoformat()}.md"

TARGETS = {
    "gateway": WORKSPACE / "memory" / "gateway.md",
    "worker": WORKSPACE / "memory" / "worker.md",
    "browser": WORKSPACE / "memory" / "browser.md",
    "preference": WORKSPACE / "memory" / "preferences.md",
    "learned-fix": WORKSPACE / "memory" / "learned-fixes.md",
    "project-progress": WORKSPACE / "memory" / "projects" / "openclaw-setup" / "progress.md",
    "project-decision": WORKSPACE / "memory" / "projects" / "openclaw-setup" / "decisions.md",
    "project-issue": WORKSPACE / "memory" / "projects" / "openclaw-setup" / "issues.md",
    "daily": None,
}

def resolve_target(kind: str) -> Path:
    if kind == "daily":
        return today_file()
    if kind not in TARGETS:
        raise ValueError(f"Unknown kind: {kind}")
    return TARGETS[kind]

def normalize_text(s: str) -> str:
    return " ".join(s.strip().split())

def target_contains_text(target: Path, text: str) -> bool:
    if not target.exists():
        return False
    wanted = normalize_text(text)
    for line in target.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            existing = normalize_text(stripped[2:])
            if existing == wanted:
                return True
    return False

def auto_kind(text: str) -> str:
    t = text.lower()

    if any(x in t for x in ["gateway", "nginx", "loopback", "18789", "bind mode", "gateway auth"]):
        return "gateway"
    if any(x in t for x in ["worker", "browserless", "upload api", "couchdb", "vnc", "tailscale ip"]):
        return "worker"
    if any(x in t for x in ["browser", "chromium", "cdp", "devtools", "9222", "9223", "9224", "visible browser", "headless browser"]):
        return "browser"
    if any(x in t for x in ["prefer", "preference", "workflow", "likes to", "wants to"]):
        return "preference"
    if any(x in t for x in ["fix", "fixed", "error", "issue", "failed", "auth", "bug", "workaround"]):
        return "learned-fix"
    if any(x in t for x in ["decision", "decided", "we chose", "we will use"]):
        return "project-decision"
    if any(x in t for x in ["progress", "done", "completed", "finished", "working now"]):
        return "project-progress"
    if any(x in t for x in ["problem", "blocker", "still broken", "not working"]):
        return "project-issue"

    return "daily"

def append_memory(target: Path, text: str, reason: str | None):
    target.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat(timespec="seconds")
    lines = [f"\n- {text.strip()}"]
    if reason:
        lines.append(f"  - reason: {reason.strip()}")
        lines.append(f"  - recorded_at: {timestamp}")
    else:
        lines.append(f"  - recorded_at: {timestamp}")

    with open(target, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def run(cmd):
    print(f"+ {' '.join(str(x) for x in cmd)}")
    result = subprocess.run(cmd, check=True)
    return result.returncode

def reconcile_memory(target: Path, text: str) -> str:
    result = subprocess.run(
        [str(VENV_PYTHON), str(RECONCILE_SCRIPT), "--target", str(target), "--text", text],
        capture_output=True,
        text=True,
        check=True,
    )
    out = result.stdout.strip()

    last = out.splitlines()[-1].strip()

    if last == "DUPLICATE":
        return "DUPLICATE"
    if last == "EVOLVE_REPLACE":
        return "EVOLVE_REPLACE"
    if last == "EVOLVE_APPEND":
        return "EVOLVE_APPEND"
    return "NEW"

def rebuild_index():
    db = WORKSPACE / ".memory-index" / "memory.sqlite"
    emb_dir = WORKSPACE / ".memory-index" / "embeddings"
    chunk_script = WORKSPACE / ".memory-index" / "scripts" / "chunk_memory.py"
    embed_script = WORKSPACE / ".memory-index" / "scripts" / "embed_memory.py"

    run([str(VENV_PYTHON), str(chunk_script)])
    run(["sqlite3", str(db), "DELETE FROM embeddings;"])
    for p in emb_dir.glob("*.npy"):
        p.unlink()
    run([str(VENV_PYTHON), str(embed_script)])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", default="auto")
    parser.add_argument("--text", required=True)
    parser.add_argument("--reason", default="")
    parser.add_argument("--no-rebuild", action="store_true")
    parser.add_argument("--allow-duplicate", action="store_true")
    args = parser.parse_args()

    kind = args.kind
    if kind == "auto":
        kind = auto_kind(args.text)

    target = resolve_target(kind)
    print(f"Resolved kind: {kind}")

    if not args.allow_duplicate and target_contains_text(target, args.text):
        print(f"SKIP exact duplicate memory in: {target}")
        return

    decision = reconcile_memory(target, args.text)
    print(f"Reconcile decision: {decision}")

    if decision == "DUPLICATE":
        print(f"SKIP semantic duplicate memory in: {target}")
        return

    if decision == "EVOLVE_REPLACE":
        text_to_write = f"[EVOLVE_REPLACE] {args.text}"
    elif decision == "EVOLVE_APPEND":
        text_to_write = f"[EVOLVE_APPEND] {args.text}"
    else:
        text_to_write = args.text

    append_memory(target, text_to_write, args.reason or None)
    print(f"Wrote memory to: {target}")

    if not args.no_rebuild:
        rebuild_index()
        print("Rebuilt chunks and embeddings.")

if __name__ == "__main__":
    main()
