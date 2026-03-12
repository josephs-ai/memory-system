import json
import subprocess
import tempfile
from pathlib import Path

WORKSPACE = Path.home() / ".openclaw" / "workspace"
REVIEW_DIR = WORKSPACE / "memory" / "review"
MEMORY_DIR = WORKSPACE / "memory"
SCRIPTS_DIR = WORKSPACE / ".memory-index" / "scripts"

APPLY_FEEDBACK = SCRIPTS_DIR / "apply_feedback_actions.py"
SYNC_MARKDOWN = SCRIPTS_DIR / "sync_registry_to_markdown.py"

CANONICAL_ITEMS = REVIEW_DIR / "canonical-items.jsonl"
CANONICAL_SUPERSEDED = REVIEW_DIR / "canonical-superseded.jsonl"
RETRIEVAL_FEEDBACK = REVIEW_DIR / "retrieval-feedback.jsonl"


def backup(path: Path):
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def restore(path: Path, content):
    if content is None:
        if path.exists():
            path.unlink()
    else:
        path.write_text(content, encoding="utf-8")


def write_jsonl(path: Path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path):
    out = []
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            out.append(json.loads(s))
    return out


def assert_true(name: str, cond: bool, detail: str = ""):
    if not cond:
        raise AssertionError(f"{name} failed. {detail}")
    print(f"PASS {name}")


def main():
    backup_items = backup(CANONICAL_ITEMS)
    backup_sup = backup(CANONICAL_SUPERSEDED)
    backup_feedback = backup(RETRIEVAL_FEEDBACK)

    browser_md = MEMORY_DIR / "browser.md"
    prefs_md = MEMORY_DIR / "preferences.md"
    memory_system_md = MEMORY_DIR / "memory-system.md"
    learned_fixes_md = MEMORY_DIR / "learned-fixes.md"

    backup_browser = backup(browser_md)
    backup_prefs = backup(prefs_md)
    backup_memory_system = backup(memory_system_md)
    backup_learned_fixes = backup(learned_fixes_md)

    try:
        canonical_items = [
            {
                "id": "active-good-1",
                "text": "Browser control works through the forwarded DevTools port.",
                "memory_type": "fact",
                "scope": "stable",
                "entity": "browser",
                "property": "tool_access_mode",
                "value": "cli_fallback",
                "status": "active",
                "target_file": "browser.md",
                "target_section": "Active",
            },
            {
                "id": "active-bad-1",
                "text": "Old worker path is the current best route.",
                "memory_type": "fact",
                "scope": "stable",
                "entity": "worker",
                "property": "upload_api_mode",
                "value": "old_path",
                "status": "active",
                "target_file": "memory-system.md",
                "target_section": "Active",
            },
        ]

        canonical_superseded = [
            {
                "id": "sup-old-1",
                "text": "Browser tool access is unavailable in-session.",
                "memory_type": "fact",
                "scope": "stable",
                "entity": "browser",
                "property": "tool_access_mode",
                "value": "unavailable",
                "status": "superseded",
                "target_file": "browser.md",
                "target_section": "Superseded",
            }
        ]

        feedback_rows = [
            {
                "selected_id": "active-good-1",
                "operator_feedback": "useful",
                "considered": [
                    {"id": "active-good-1", "score": 200},
                    {"id": "sup-old-1", "score": 120},
                ],
            },
            {
                "selected_id": "active-good-1",
                "operator_feedback": "useful",
                "considered": [
                    {"id": "active-good-1", "score": 205},
                    {"id": "sup-old-1", "score": 110},
                ],
            },
            {
                "selected_id": "active-bad-1",
                "operator_feedback": "bad",
                "considered": [
                    {"id": "active-bad-1", "score": 180},
                ],
            },
            {
                "selected_id": "active-bad-1",
                "operator_feedback": "bad",
                "considered": [
                    {"id": "active-bad-1", "score": 175},
                ],
            },
            {
                "selected_id": "active-bad-1",
                "operator_feedback": "bad",
                "considered": [
                    {"id": "active-bad-1", "score": 170},
                ],
            },
            {
                "selected_id": "active-good-1",
                "considered": [
                    {"id": "sup-old-1", "score": 100},
                ],
            },
        ]

        write_jsonl(CANONICAL_ITEMS, canonical_items)
        write_jsonl(CANONICAL_SUPERSEDED, canonical_superseded)
        write_jsonl(RETRIEVAL_FEEDBACK, feedback_rows)

        subprocess.run(
            ["python3", str(APPLY_FEEDBACK)],
            check=True,
            capture_output=True,
            text=True,
        )

        items_after = load_jsonl(CANONICAL_ITEMS)
        sup_after = load_jsonl(CANONICAL_SUPERSEDED)

        good_item = next(x for x in items_after if x["id"] == "active-good-1")
        bad_item = next(x for x in items_after if x["id"] == "active-bad-1")
        sup_item = next(x for x in sup_after if x["id"] == "sup-old-1")

        assert_true(
            "useful_feedback_bonus_applied",
            int(good_item.get("ranking_bonus", 0)) > 0,
            str(good_item),
        )
        assert_true(
            "bad_feedback_penalty_applied",
            int(bad_item.get("ranking_penalty", 0)) > 0,
            str(bad_item),
        )
        assert_true(
            "superseded_penalty_applied",
            int(sup_item.get("ranking_penalty", 0)) > 0,
            str(sup_item),
        )

        subprocess.run(
            ["python3", str(SYNC_MARKDOWN)],
            check=True,
            capture_output=True,
            text=True,
        )

        browser_text = browser_md.read_text(encoding="utf-8")
        memory_system_text = memory_system_md.read_text(encoding="utf-8")

        assert_true(
            "browser_md_has_active",
            "Browser control works through the forwarded DevTools port." in browser_text,
            browser_text,
        )
        assert_true(
            "browser_md_has_superseded",
            "Browser tool access is unavailable in-session." in browser_text,
            browser_text,
        )
        assert_true(
            "memory_system_md_has_active",
            "Old worker path is the current best route." in memory_system_text,
            memory_system_text,
        )

        print()
        print("ALL AUTOMATION TESTS PASSED")

    finally:
        restore(CANONICAL_ITEMS, backup_items)
        restore(CANONICAL_SUPERSEDED, backup_sup)
        restore(RETRIEVAL_FEEDBACK, backup_feedback)
        restore(browser_md, backup_browser)
        restore(prefs_md, backup_prefs)
        restore(memory_system_md, backup_memory_system)
        restore(learned_fixes_md, backup_learned_fixes)


if __name__ == "__main__":
    main()
