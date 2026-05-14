"""
Create project scaffold.

Key functions: main
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from memory_db import create_project_registry_row, close_pool

WORKSPACE = Path.home() / ".openclaw" / "workspace"
PROJECTS_DIR = WORKSPACE / "memory" / "projects"

STOP_WATCHER = SCRIPT_DIR / "stop_tracked_path_watcher.py"
START_WATCHER = SCRIPT_DIR / "start_tracked_path_watcher.py"

SUBDIRS = [
    "active",
    "canonical",
    "chunks",
    "dehydrated",
    "routing",
    "working",
    "summaries",
    "sessions",
    "subprojects",
    "archived",
    "local",
    "config",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a manual parent-project scaffold")
    parser.add_argument("--project-id", required=True, help="Unique ID for the project")
    parser.add_argument("--name", required=True, help="Human-readable name")
    parser.add_argument("--type", choices=["parent", "subproject"], default="parent")
    parser.add_argument("--parent-id", default=None, help="Parent ID if this is a subproject")
    parser.add_argument("--workspace-mode", choices=["standard", "unreal", "unity", "godot"], default="standard")
    args = parser.parse_args()

    if args.type == "subproject":
        if not args.parent_id:
            print("Error: --parent-id is required for subproject scaffolds.")
            sys.exit(1)
        parent_root = PROJECTS_DIR / args.parent_id
        if not parent_root.exists():
            print(f"Error: Parent project directory {parent_root} does not exist.")
            sys.exit(1)
        project_root = parent_root / "subprojects" / args.project_id
    else:
        project_root = PROJECTS_DIR / args.project_id

    active_root = project_root / "active"

    print(f"Creating project scaffold for: {args.project_id} (type: {args.type})")

    dir_existed = project_root.exists()
    if dir_existed:
        print(f"Notice: Project directory {project_root} already exists. Ensuring subdirectories and re-registering.")

    # Create the filesystem scaffold atomically
    try:
        project_root.mkdir(parents=True, exist_ok=True)
        for subdir in SUBDIRS:
            (project_root / subdir).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Failed to ensure directory structure: {e}")
        if not dir_existed:
            shutil.rmtree(project_root, ignore_errors=True)
        sys.exit(1)

    print("Directories ensured. Registering project in database...")

    now = time.time()
    project_record = {
        "project_id": args.project_id,
        "name": args.name,
        "type": args.type,
        "parent_project_id": args.parent_id,
        "status": "active",
        "memory_root": str(project_root),
        "active_memory_root": str(active_root),
        "archived_memory_scope": "global",
        "allow_model_subprojects": 0,
        "created_by": "manual",
        "workspace_mode": args.workspace_mode,
        "created_at": now,
        "updated_at": now,
    }

    tracked_paths = [
        {
            "project_id": args.project_id,
            "path": str(active_root),
            "kind": "active_root",
            "created_at": now,
        }
    ]

    try:
        create_project_registry_row(project_record, tracked_paths)
    except Exception as e:
        print(f"Database insertion failed: {e}")
        if not dir_existed:
            print("Rolling back: Deleting orphaned filesystem structure...")
            shutil.rmtree(project_root, ignore_errors=True)
        close_pool()
        sys.exit(1)

    print("Project registered successfully.")

    # Trigger a restart of the tracked path watcher
    print("Restarting tracked path watcher to apply new paths...")
    if STOP_WATCHER.exists() and START_WATCHER.exists():
        try:
            subprocess.run([sys.executable or "python3", str(STOP_WATCHER)], check=False, timeout=10)
            time.sleep(1.0)
            subprocess.run([sys.executable or "python3", str(START_WATCHER)], check=True, timeout=10)
            print("Watcher restarted.")
        except subprocess.TimeoutExpired:
            print("Warning: Watcher restart timed out. It may need a manual restart.")
        except Exception as e:
            print(f"Warning: Watcher restart failed: {e}. It may need a manual restart.")
    else:
        print("Warning: Watcher start/stop scripts not found. Watcher may need a manual restart.")

    close_pool()
    print("Scaffold complete.")


if __name__ == "__main__":
    main()
