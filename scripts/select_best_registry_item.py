import json
import argparse
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime, timezone

from memory_db import fetch_memory_items, feedback_stats_db, close_pool

WORKSPACE = Path.home() / ".openclaw" / "workspace"
SCRIPTS_DIR = WORKSPACE / ".memory-index" / "scripts"
MATCH_SCRIPT = SCRIPTS_DIR / "match_memory_items.py"


def json_safe(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


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


def state_bonus(item: dict) -> int:
    status = (item.get("status") or "active").strip().lower()

    if status == "active":
        return 30
    if status == "uncertain":
        return -15
    if status == "superseded":
        return -60
    if status == "archived":
        return -90
    return 0


def confidence_bonus(item: dict) -> int:
    try:
        conf = float(item.get("confidence", 1.0) or 1.0)
    except Exception:
        conf = 1.0

    if conf >= 0.95:
        return 20
    if conf >= 0.90:
        return 14
    if conf >= 0.80:
        return 8
    if conf >= 0.65:
        return 2
    return -8


def freshness_class_bonus(item: dict) -> int:
    fc = (item.get("freshness_class") or "").strip().lower()

    if fc == "permanent":
        return 8
    if fc == "long_lived":
        return 4
    if fc == "volatile":
        return 0
    return 0


def parse_iso(ts):
    if not ts:
        return None

    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts

    try:
        parsed = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed
    except Exception:
        return None


def freshness_bonus(item: dict) -> int:
    ts = parse_iso(item.get("last_confirmed"))
    if ts is None:
        return 0

    now = datetime.now(timezone.utc)
    age_days = (now - ts).days

    if age_days <= 3:
        return 20
    if age_days <= 14:
        return 12
    if age_days <= 45:
        return 6
    if age_days <= 120:
        return 0
    return -8


def manual_ranking_bonus(item: dict) -> int:
    try:
        return int(float(item.get("ranking_bonus", 0) or 0))
    except Exception:
        return 0


def manual_ranking_penalty(item: dict) -> int:
    try:
        return int(float(item.get("ranking_penalty", 0) or 0))
    except Exception:
        return 0


def feedback_bonus(item: dict) -> int:
    item_id = item.get("id")
    if not item_id:
        return 0

    useful, bad = feedback_stats_db(item_id)
    return (useful * 6) - (bad * 10)


def total_score(match: dict, item: dict) -> int:
    return (
        score_match(match)
        + state_bonus(item)
        + freshness_bonus(item)
        + confidence_bonus(item)
        + freshness_class_bonus(item)
        + feedback_bonus(item)
        + manual_ranking_bonus(item)
        - manual_ranking_penalty(item)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--target-file", required=True)
    parser.add_argument("--include-superseded", action="store_true")
    args = parser.parse_args()

    statuses = ["active"]
    if args.include_superseded:
        statuses.append("superseded")

    items = fetch_memory_items(statuses)
    candidate_path = Path(args.candidate)

    registry_items = [
        item for item in items
        if item.get("target_file") == args.target_file
    ]

    if not registry_items:
        print("NONE")
        close_pool()
        return

    best = None
    best_score = -10**9
    best_match = None

    for item in registry_items:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
            json.dump(item, tmp, ensure_ascii=False, indent=2, default=json_safe)
            tmp_path = tmp.name

        try:
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
                check=True,
            ).stdout

            match = json.loads(match_out)
            score = total_score(match, item)

            if score > best_score:
                best_score = score
                best = item
                best_match = match

        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

    if best is None:
        print("NONE")
        close_pool()
        return

    useful_count, bad_count = feedback_stats_db(best.get("id"))

    out = {
        "score": best_score,
        "existing_item": best,
        "match": best_match,
        "freshness_bonus": freshness_bonus(best),
        "state_bonus": state_bonus(best),
        "confidence_bonus": confidence_bonus(best),
        "freshness_class_bonus": freshness_class_bonus(best),
        "manual_ranking_bonus": manual_ranking_bonus(best),
        "manual_ranking_penalty": manual_ranking_penalty(best),
        "feedback_bonus": feedback_bonus(best),
        "feedback_stats": {
            "useful": useful_count,
            "bad": bad_count,
        },
    }

    print(json.dumps(out, ensure_ascii=False, indent=2, default=json_safe))
    close_pool()


if __name__ == "__main__":
    main()
