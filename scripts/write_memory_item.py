"""
Memory system utility: write memory item.

Key functions: load_json, json_safe, target_file_for_item, write_temp_item
"""
import json
import argparse
import subprocess
import tempfile
import sys
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from memory_db import fetch_memory_items, upsert_memory_item, close_pool

WORKSPACE = Path.home() / ".openclaw" / "workspace"
SCRIPTS_DIR = WORKSPACE / ".memory-index" / "scripts"
MEMORY_DIR = WORKSPACE / "memory"

RECONCILE_SCRIPT = SCRIPTS_DIR / "reconcile_memory_items.py"
PATCH_SCRIPT = SCRIPTS_DIR / "patch_memory_file.py"
SELECT_REGISTRY_SCRIPT = SCRIPTS_DIR / "select_best_registry_item.py"


def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_safe(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def target_file_for_item(item: dict) -> Path:
    memory_type = item.get("memory_type")
    entity = item.get("entity")
    scope = item.get("scope")

    if entity == "checkpoint_pipeline":
        return MEMORY_DIR / "memory-system.md"

    if memory_type == "learned_fix":
        return MEMORY_DIR / "learned-fixes.md"

    if entity == "browser":
        return MEMORY_DIR / "browser.md"

    if entity == "gateway":
        return MEMORY_DIR / "gateway.md"

    if entity == "worker":
        return MEMORY_DIR / "worker.md"

    if scope == "daily":
        return MEMORY_DIR / "daily" / f"{datetime.now().date().isoformat()}.md"

    return MEMORY_DIR / "projects.md"


def write_temp_item(item: dict) -> Path:
    tmp = Path(tempfile.mkstemp(suffix=".json")[1])
    tmp.write_text(json.dumps(item, ensure_ascii=False, indent=2, default=json_safe), encoding="utf-8")
    return tmp


def registry_target_name(target: Path) -> str:
    return target.name


def fetch_active_registry_items():
    return fetch_memory_items(["active"])


def upsert_final_item(item: dict, target_name: str):
    item = dict(item)
    item["target_file"] = target_name
    item.setdefault("target_section", "Active")
    item["status"] = "active"

    if item.get("suggested_route") in {"auto", "inbox", "daily", "project", "discard"}:
        item["suggested_route"] = None

    upsert_memory_item(item)


def mark_item_superseded(existing_item: dict):
    updated = dict(existing_item)
    updated["status"] = "superseded"
    upsert_memory_item(updated)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-item", required=True)
    args = parser.parse_args()

    candidate = load_json(args.candidate_item)
    target = target_file_for_item(candidate)
    target_name = registry_target_name(target)

    if not target.exists():
        print(f"ERROR target file does not exist: {target}")
        close_pool()
        return

    best_out = subprocess.run(
        [
            "python3",
            str(SELECT_REGISTRY_SCRIPT),
            "--candidate",
            args.candidate_item,
            "--target-file",
            target_name,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    if not best_out or best_out == "NONE":
        patch = subprocess.run(
            [
                "python3",
                str(PATCH_SCRIPT),
                "--target",
                str(target),
                "--outcome",
                "APPEND_NEW",
                "--candidate-item",
                args.candidate_item,
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        upsert_final_item(candidate, target_name)

        print(f"target={target}")
        print("outcome=APPEND_NEW")
        print(patch.stdout.strip())
        close_pool()
        return

    best_obj = json.loads(best_out)
    existing_item = best_obj["existing_item"]
    score = best_obj["score"]

    if score < 35:
        patch = subprocess.run(
            [
                "python3",
                str(PATCH_SCRIPT),
                "--target",
                str(target),
                "--outcome",
                "APPEND_NEW",
                "--candidate-item",
                args.candidate_item,
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        upsert_final_item(candidate, target_name)

        print(f"target={target}")
        print("outcome=APPEND_NEW")
        print(f"best_score={score}")
        print(patch.stdout.strip())
        close_pool()
        return

    existing_tmp = write_temp_item(existing_item)

    try:
        rec = subprocess.run(
            [
                "python3",
                str(RECONCILE_SCRIPT),
                "--candidate",
                args.candidate_item,
                "--existing",
                str(existing_tmp),
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        rec_obj = json.loads(rec.stdout)
        outcome = rec_obj["outcome"]

        patch_cmd = [
            "python3",
            str(PATCH_SCRIPT),
            "--target",
            str(target),
            "--outcome",
            outcome,
            "--candidate-item",
            args.candidate_item,
        ]

        if outcome in {"MERGE_INTO_EXISTING", "SUPERSEDE_EXISTING"}:
            patch_cmd += ["--existing-item", str(existing_tmp)]

        patch = subprocess.run(
            patch_cmd,
            capture_output=True,
            text=True,
            check=True,
        )

        if outcome == "IGNORE_DUPLICATE":
            print(f"target={target}")
            print(f"best_score={score}")
            print(f"matched_existing_id={existing_item.get('id')}")
            print("outcome=IGNORE_DUPLICATE")
            print("NO_CHANGE duplicate")
            close_pool()
            return

        if outcome == "MERGE_INTO_EXISTING":
            merged = dict(existing_item)
            merged.update(candidate)
            merged["id"] = existing_item.get("id")
            merged["target_file"] = target_name
            merged["target_section"] = existing_item.get("target_section") or "Active"
            merged["status"] = "active"
            if merged.get("suggested_route") in {"auto", "inbox", "daily", "project", "discard"}:
                merged["suggested_route"] = None
            upsert_memory_item(merged)

        elif outcome == "SUPERSEDE_EXISTING":
            mark_item_superseded(existing_item)
            new_item = dict(candidate)
            new_item["target_file"] = target_name
            new_item["target_section"] = "Active"
            new_item["status"] = "active"
            new_item["supersedes"] = existing_item.get("id")
            if new_item.get("suggested_route") in {"auto", "inbox", "daily", "project", "discard"}:
                new_item["suggested_route"] = None
            upsert_memory_item(new_item)

        elif outcome == "APPEND_NEW":
            upsert_final_item(candidate, target_name)

        print(f"target={target}")
        print(f"best_score={score}")
        print(f"matched_existing_id={existing_item.get('id')}")
        print(f"outcome={outcome}")
        print(patch.stdout.strip())

    finally:
        try:
            existing_tmp.unlink()
        except Exception:
            pass
        close_pool()


if __name__ == "__main__":
    main()
