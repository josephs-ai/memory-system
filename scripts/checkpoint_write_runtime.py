"""
Checkpoint write runtime state for crash recovery.

Key functions: load_json, json_safe, target_file_for_item, write_temp_item
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from memory_db import fetch_memory_items, upsert_memory_item

WORKSPACE = Path.home() / ".openclaw" / "workspace"
SCRIPTS_DIR = WORKSPACE / ".memory-index" / "scripts"
MEMORY_DIR = WORKSPACE / "memory"

RECONCILE_SCRIPT = SCRIPTS_DIR / "reconcile_memory_items.py"
PATCH_SCRIPT = SCRIPTS_DIR / "patch_memory_file.py"
SELECT_REGISTRY_SCRIPT = SCRIPTS_DIR / "select_best_registry_item.py"


def load_json(path: str) -> dict:
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
    if memory_type == "learned_fix" or memory_type == "lesson_learned":
        return MEMORY_DIR / "learned-fixes.md"
    if memory_type == "architecture_rule" or memory_type == "rule" or memory_type == "implementation_pattern":
        return MEMORY_DIR / "architecture.md"
    if memory_type == "decision":
        return MEMORY_DIR / "decisions.md"
    if memory_type == "observation":
        return MEMORY_DIR / "observations.md"
    if memory_type == "preference":
        return MEMORY_DIR / "preferences.md"
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
    final_item = dict(item)
    final_item["target_file"] = target_name
    final_item.setdefault("target_section", "Active")
    final_item["status"] = "active"

    if final_item.get("suggested_route") in {"auto", "inbox", "daily", "project", "discard"}:
        final_item["suggested_route"] = None

    upsert_memory_item(final_item)


def mark_item_superseded(existing_item: dict):
    updated = dict(existing_item)
    updated["status"] = "superseded"
    upsert_memory_item(updated)


def run_capture(cmd: list[str]) -> str:
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def apply_candidate_item(candidate_item_path: str) -> str:
    candidate = load_json(candidate_item_path)
    target = target_file_for_item(candidate)
    target_name = registry_target_name(target)

    if not target.exists():
        raise FileNotFoundError(f"target file does not exist: {target}")

    best_out = run_capture(
        [
            "python3",
            str(SELECT_REGISTRY_SCRIPT),
            "--candidate",
            candidate_item_path,
            "--target-file",
            target_name,
        ]
    )

    if not best_out or best_out == "NONE":
        patch_out = run_capture(
            [
                "python3",
                str(PATCH_SCRIPT),
                "--target",
                str(target),
                "--outcome",
                "APPEND_NEW",
                "--candidate-item",
                candidate_item_path,
            ]
        )
        upsert_final_item(candidate, target_name)
        return "\n".join(
            [
                f"target={target}",
                "outcome=APPEND_NEW",
                patch_out,
            ]
        ).strip()

    best_obj = json.loads(best_out)
    existing_item = best_obj["existing_item"]
    score = best_obj["score"]

    if score < 35:
        patch_out = run_capture(
            [
                "python3",
                str(PATCH_SCRIPT),
                "--target",
                str(target),
                "--outcome",
                "APPEND_NEW",
                "--candidate-item",
                candidate_item_path,
            ]
        )
        upsert_final_item(candidate, target_name)
        return "\n".join(
            [
                f"target={target}",
                "outcome=APPEND_NEW",
                f"best_score={score}",
                patch_out,
            ]
        ).strip()

    existing_tmp = write_temp_item(existing_item)
    try:
        rec_out = run_capture(
            [
                "python3",
                str(RECONCILE_SCRIPT),
                "--candidate",
                candidate_item_path,
                "--existing",
                str(existing_tmp),
            ]
        )
        rec_obj = json.loads(rec_out)
        outcome = rec_obj["outcome"]

        patch_cmd = [
            "python3",
            str(PATCH_SCRIPT),
            "--target",
            str(target),
            "--outcome",
            outcome,
            "--candidate-item",
            candidate_item_path,
        ]
        if outcome in {"MERGE_INTO_EXISTING", "SUPERSEDE_EXISTING"}:
            patch_cmd += ["--existing-item", str(existing_tmp)]

        patch_out = run_capture(patch_cmd)

        if outcome == "IGNORE_DUPLICATE":
            return "\n".join(
                [
                    f"target={target}",
                    f"best_score={score}",
                    f"matched_existing_id={existing_item.get('id')}",
                    "outcome=IGNORE_DUPLICATE",
                    "NO_CHANGE duplicate",
                ]
            ).strip()

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

        return "\n".join(
            [
                f"target={target}",
                f"best_score={score}",
                f"matched_existing_id={existing_item.get('id')}",
                f"outcome={outcome}",
                patch_out,
            ]
        ).strip()
    finally:
        try:
            existing_tmp.unlink()
        except Exception:
            pass
