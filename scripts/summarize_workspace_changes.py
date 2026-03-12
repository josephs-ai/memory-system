import argparse
import json
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def add_summary(out, scope, category, text, evidence):
    out.append({
        "scope": scope,
        "category": category,
        "text": text,
        "evidence": evidence,
    })


def classify_paths(paths):
    buckets = {
        "memory_system": [],
        "browser": [],
        "worker": [],
        "gateway": [],
        "docs": [],
        "project_state": [],
        "other": [],
    }

    for p in paths:
        s = p.lower()

        if ".memory-index" in s or "/memory/" in s or s.startswith("memory/") or "memory" in s:
            buckets["memory_system"].append(p)
        elif "browser" in s or "cdp" in s or "devtools" in s or "vnc" in s:
            buckets["browser"].append(p)
        elif "worker" in s or "upload" in s or "couchdb" in s:
            buckets["worker"].append(p)
        elif "gateway" in s or "proxy" in s:
            buckets["gateway"].append(p)
        elif s.endswith(".md") or "boot" in s or "identity" in s or "tools" in s or "user" in s:
            buckets["docs"].append(p)
        elif "project-state" in s or "knowledge_base" in s or "project" in s:
            buckets["project_state"].append(p)
        else:
            buckets["other"].append(p)

    return buckets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    args = parser.parse_args()

    snapshot_path = Path(args.snapshot).expanduser().resolve()
    snap = load_json(snapshot_path)

    changed = snap.get("changed_files", [])
    added = snap.get("added_files", [])
    removed = snap.get("removed_files", [])
    modified = snap.get("modified_files", [])
    untracked = snap.get("untracked_files", [])
    top_dirs = snap.get("top_directories_touched", [])

    buckets = classify_paths(changed)
    summaries = []

    if buckets["memory_system"]:
        add_summary(
            summaries,
            "project",
            "memory_system",
            f"Memory-system/project infrastructure files changed in {len(buckets['memory_system'])} path(s).",
            buckets["memory_system"][:10],
        )

    if buckets["browser"]:
        add_summary(
            summaries,
            "project",
            "browser",
            f"Browser-related files changed in {len(buckets['browser'])} path(s).",
            buckets["browser"][:10],
        )

    if buckets["worker"]:
        add_summary(
            summaries,
            "project",
            "worker",
            f"Worker/upload-related files changed in {len(buckets['worker'])} path(s).",
            buckets["worker"][:10],
        )

    if buckets["gateway"]:
        add_summary(
            summaries,
            "project",
            "gateway",
            f"Gateway/proxy-related files changed in {len(buckets['gateway'])} path(s).",
            buckets["gateway"][:10],
        )

    if buckets["docs"]:
        add_summary(
            summaries,
            "daily",
            "docs",
            f"Documentation/instruction files changed in {len(buckets['docs'])} path(s).",
            buckets["docs"][:10],
        )

    if untracked:
        add_summary(
            summaries,
            "daily",
            "untracked",
            f"{len(untracked)} untracked file/path entries are present in the workspace snapshot.",
            untracked[:15],
        )

    if added:
        add_summary(
            summaries,
            "daily",
            "added",
            f"{len(added)} added file(s) detected.",
            added[:15],
        )

    if removed:
        add_summary(
            summaries,
            "daily",
            "removed",
            f"{len(removed)} removed file(s) detected.",
            removed[:15],
        )

    if modified:
        add_summary(
            summaries,
            "daily",
            "modified",
            f"{len(modified)} modified file(s) detected.",
            modified[:15],
        )

    if top_dirs:
        dirs = [f"{x['dir']} ({x['count']})" for x in top_dirs[:8]]
        add_summary(
            summaries,
            "daily",
            "top_dirs",
            "Top directories/files touched: " + ", ".join(dirs),
            dirs,
        )

    out = {
        "captured_at": snap.get("captured_at"),
        "workspace": snap.get("workspace"),
        "summary_count": len(summaries),
        "summaries": summaries,
    }

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
