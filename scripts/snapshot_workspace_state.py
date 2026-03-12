import argparse
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run_capture(cmd, cwd=None):
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def is_git_repo(workspace: Path) -> bool:
    code, out, _ = run_capture(["git", "rev-parse", "--is-inside-work-tree"], cwd=workspace)
    return code == 0 and out.lower() == "true"


def parse_git_status_porcelain(workspace: Path):
    code, out, err = run_capture(["git", "status", "--porcelain"], cwd=workspace)
    if code != 0:
        return {
            "tracked_changed": [],
            "added_files": [],
            "removed_files": [],
            "modified_files": [],
            "renamed_files": [],
            "untracked_files": [],
            "raw_lines": [],
            "error": err,
        }

    tracked_changed = []
    added_files = []
    removed_files = []
    modified_files = []
    renamed_files = []
    untracked_files = []
    raw_lines = out.splitlines()

    for line in raw_lines:
        if not line.strip():
            continue

        if line.startswith("?? "):
            path = line[3:].strip()
            untracked_files.append(path)
            continue

        status = line[:2]
        path = line[3:].strip()

        tracked_changed.append(path)

        if "A" in status:
            added_files.append(path)
        if "D" in status:
            removed_files.append(path)
        if "M" in status:
            modified_files.append(path)
        if "R" in status:
            renamed_files.append(path)

    return {
        "tracked_changed": sorted(set(tracked_changed)),
        "added_files": sorted(set(added_files)),
        "removed_files": sorted(set(removed_files)),
        "modified_files": sorted(set(modified_files)),
        "renamed_files": sorted(set(renamed_files)),
        "untracked_files": sorted(set(untracked_files)),
        "raw_lines": raw_lines,
        "error": "",
    }


def get_recent_commit_summary(workspace: Path):
    code, out, err = run_capture(
        ["git", "log", "-1", "--pretty=format:%H%n%s%n%ci"],
        cwd=workspace,
    )
    if code != 0 or not out:
        return {"commit": None, "subject": None, "committed_at": None, "error": err}

    lines = out.splitlines()
    return {
        "commit": lines[0] if len(lines) > 0 else None,
        "subject": lines[1] if len(lines) > 1 else None,
        "committed_at": lines[2] if len(lines) > 2 else None,
        "error": "",
    }


def top_dirs_from_paths(paths):
    counter = Counter()
    for p in paths:
        parts = Path(p).parts
        if not parts:
            continue
        top = parts[0]
        counter[top] += 1
    return [{"dir": name, "count": count} for name, count in counter.most_common()]


def fallback_recent_files(workspace: Path, recent_minutes: int):
    now_ts = datetime.now(timezone.utc).timestamp()
    recent_seconds = recent_minutes * 60
    touched = []

    skip_dirs = {
        ".git",
        "__pycache__",
        ".venv",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
    }

    for path in workspace.rglob("*"):
        if not path.is_file():
            continue

        if any(part in skip_dirs for part in path.parts):
            continue

        try:
            mtime = path.stat().st_mtime
        except Exception:
            continue

        if (now_ts - mtime) <= recent_seconds:
            touched.append(str(path.relative_to(workspace)))

    return sorted(touched)


def build_snapshot(workspace: Path, recent_minutes: int):
    snapshot = {
        "captured_at": now_iso(),
        "workspace": str(workspace),
        "is_git_repo": False,
        "changed_files": [],
        "added_files": [],
        "removed_files": [],
        "modified_files": [],
        "renamed_files": [],
        "untracked_files": [],
        "top_directories_touched": [],
        "git_status_summary": {},
        "recent_commit": {},
        "fallback_recent_files": [],
    }

    if is_git_repo(workspace):
        snapshot["is_git_repo"] = True
        status = parse_git_status_porcelain(workspace)
        changed_files = sorted(
            set(
                status["tracked_changed"]
                + status["untracked_files"]
            )
        )

        snapshot["changed_files"] = changed_files
        snapshot["added_files"] = status["added_files"]
        snapshot["removed_files"] = status["removed_files"]
        snapshot["modified_files"] = status["modified_files"]
        snapshot["renamed_files"] = status["renamed_files"]
        snapshot["untracked_files"] = status["untracked_files"]
        snapshot["top_directories_touched"] = top_dirs_from_paths(changed_files)
        snapshot["git_status_summary"] = {
            "tracked_changed_count": len(status["tracked_changed"]),
            "untracked_count": len(status["untracked_files"]),
            "raw_status_lines": status["raw_lines"],
        }
        snapshot["recent_commit"] = get_recent_commit_summary(workspace)
    else:
        recent_files = fallback_recent_files(workspace, recent_minutes)
        snapshot["fallback_recent_files"] = recent_files
        snapshot["changed_files"] = recent_files
        snapshot["top_directories_touched"] = top_dirs_from_paths(recent_files)

    return snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--recent-minutes", type=int, default=120)
    parser.add_argument("--out")
    args = parser.parse_args()

    workspace = Path(args.workspace).expanduser().resolve()
    snapshot = build_snapshot(workspace, args.recent_minutes)

    text = json.dumps(snapshot, ensure_ascii=False, indent=2)

    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(args.out)
    else:
        print(text)


if __name__ == "__main__":
    main()
