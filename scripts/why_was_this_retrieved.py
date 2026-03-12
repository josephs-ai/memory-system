import json
import argparse
import subprocess
import tempfile
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from memory_db import close_pool, fetch_memory_items

WORKSPACE = Path.home() / ".openclaw" / "workspace"
SCRIPTS_DIR = WORKSPACE / ".memory-index" / "scripts"

MATCH_SCRIPT = SCRIPTS_DIR / "match_memory_items.py"


def json_safe(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_tmp(obj):
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(obj, tmp, ensure_ascii=False, indent=2, default=json_safe)
    tmp.close()
    return tmp.name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--item-id", required=True)
    args = parser.parse_args()

    items = fetch_memory_items(["active", "superseded", "archived", "uncertain"])
    target = None
    for item in items:
        if item.get("id") == args.item_id:
            target = item
            break

    if target is None:
        print("NOT FOUND")
        close_pool()
        return

    tpath = write_tmp(target)

    try:
        out = subprocess.run(
            [
                "python3",
                str(MATCH_SCRIPT),
                "--candidate",
                args.candidate,
                "--existing",
                tpath,
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        print("=== WHY THIS WAS RETRIEVED ===")
        print()
        print("item:")
        print(json.dumps(target, ensure_ascii=False, indent=2, default=json_safe))
        print()
        print("match:")
        print(out.stdout.strip())

    finally:
        try:
            Path(tpath).unlink()
        except Exception:
            pass
        close_pool()


if __name__ == "__main__":
    main()
