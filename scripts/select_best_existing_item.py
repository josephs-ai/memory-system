"""
Select the best existing item using scoring heuristics.

Key functions: load_json, score_match, main
"""
import json
import argparse
import subprocess
import tempfile
from pathlib import Path

WORKSPACE = Path.home() / ".openclaw" / "workspace"
SCRIPTS_DIR = WORKSPACE / ".memory-index" / "scripts"
MATCH_SCRIPT = SCRIPTS_DIR / "match_memory_items.py"
EXTRACT_ACTIVE_SCRIPT = SCRIPTS_DIR / "extract_active_memory_items.py"

def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def score_match(match: dict) -> int:
    score = 0
    if match.get("same_type"):
        score += 10
    if match.get("same_entity"):
        score += 25
    if match.get("same_scope"):
        score += 10
    if match.get("same_property"):
        score += 30
    if match.get("same_value"):
        score += 40
    if match.get("comparable_identity"):
        score += 40
    if match.get("same_slot_different_value"):
        score += 35
    if match.get("weak_related"):
        score += 15
    return score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()

    candidate_path = Path(args.candidate)
    target_path = Path(args.target)

    extracted = subprocess.run(
        [
            "python3",
            str(EXTRACT_ACTIVE_SCRIPT),
            "--target",
            str(target_path),
        ],
        capture_output=True,
        text=True,
        check=True
    ).stdout.strip()

    if not extracted or extracted == "NONE":
        print("NONE")
        return

    best = None
    best_score = -1
    best_match = None

    for line in extracted.splitlines():
        s = line.strip()
        if not s:
            continue
        item = json.loads(s)

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
            json.dump(item, tmp, ensure_ascii=False, indent=2)
            tmp_path = tmp.name

        match_out = subprocess.run(
            [
                "python3",
                str(MATCH_SCRIPT),
                "--candidate",
                str(candidate_path),
                "--existing",
                tmp_path,
            ],
            capture_output=True,
            text=True,
            check=True
        ).stdout

        match = json.loads(match_out)
        score = score_match(match)

        if score > best_score:
            best_score = score
            best = item
            best_match = match

    if best is None:
        print("NONE")
        return

    out = {
        "score": best_score,
        "existing_item": best,
        "match": best_match
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
