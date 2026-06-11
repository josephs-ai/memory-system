"""
Reconcile memory items between sources — detect and resolve conflicts
between file-based and database-stored memory.
"""
import json
import argparse
import subprocess
from pathlib import Path

WORKSPACE = Path.home() / ".openclaw" / "workspace"
SCRIPTS_DIR = WORKSPACE / ".memory-index" / "scripts"
MATCH_SCRIPT = SCRIPTS_DIR / "match_memory_items.py"

SAFE_SUPERSEDE_PROPERTIES = {
    "port",
    "mode",
    "policy",
    "status",
    "model",
    "path",
    "tool_access_mode",
    "checkpoint_policy",
    "forwarded_devtools_port",
    "local_devtools_port",
    "canonical_memory_backend",
    "processing_capability",
    "build_priority",
}

ARCHIVABLE_STATUSES = {"superseded", "archived"}
LOW_CONFIDENCE_THRESHOLD = 0.75
HIGH_CONFIDENCE_THRESHOLD = 0.90


def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def can_auto_supersede(candidate: dict, existing: dict) -> bool:
    """Check if a candidate can auto-supersede an existing item.

    Default: any item sharing the same entity+property+scope can supersede
    if the candidate is newer. The old SAFE_SUPERSEDE_PROPERTIES allowlist
    was too restrictive — it only covered 14 system properties, missing
    real-world facts like location, job, theme, etc.

    Exclusions: generic/accumulative properties where multiple values are
    expected (utterance, fact, observation, episodic, tag).
    """
    NEVER_SUPERSEDE = {"utterance", "fact", "observation", "episodic", "tag", "note"}

    prop = (candidate.get("property") or "").strip().lower()
    if not prop or prop in NEVER_SUPERSEDE:
        return False

    # Both must share the same entity for supersede to apply
    c_entity = (candidate.get("entity") or "").strip().lower()
    e_entity = (existing.get("entity") or "").strip().lower()
    if not c_entity or c_entity != e_entity:
        return False

    # Candidate must be newer (or same age) to supersede
    c_ts = candidate.get("last_confirmed") or candidate.get("first_seen") or ""
    e_ts = existing.get("last_confirmed") or existing.get("first_seen") or ""
    if str(c_ts) < str(e_ts):
        return False  # Candidate is older — don't supersede

    return True

def confidence_of(item: dict) -> float:
    try:
        if "confidence" not in item or item.get("confidence") is None:
            return 1.0
        return float(item.get("confidence"))
    except Exception:
        return 1.0


def normalized_status(item: dict) -> str:
    return (item.get("status") or "active").strip().lower()


def has_missing_core_value(item: dict) -> bool:
    prop = item.get("property")
    value = item.get("value")
    if prop and not value:
        return True
    return False


def same_slot(match: dict) -> bool:
    return bool(
        match.get("same_entity")
        and match.get("same_scope")
        and match.get("same_property")
    )


def same_identity(match: dict) -> bool:
    return bool(match.get("comparable_identity"))


def classify_outcome(candidate: dict, existing: dict, match: dict) -> str:
    candidate_conf = confidence_of(candidate)
    existing_conf = confidence_of(existing)

    existing_status = normalized_status(existing)
    candidate_status = normalized_status(candidate)

    # 1) Exact same identity and value -> duplicate
    if match.get("comparable_identity") and match.get("exact_identity_and_value"):
        return "IGNORE_DUPLICATE"

    # 2) Existing already superseded/archived in same slot -> archive old lineage
    if same_slot(match) and existing_status in ARCHIVABLE_STATUSES:
        return "ARCHIVE_OLD"

    # 3) Same slot but different value
    if match.get("same_slot_different_value"):
        # weak or incomplete candidate update -> uncertain, not hard conflict
        if candidate_conf < LOW_CONFIDENCE_THRESHOLD or has_missing_core_value(candidate):
            return "MARK_UNCERTAIN"

        # safe property and strong candidate -> supersede
        if can_auto_supersede(candidate, existing):
            return "SUPERSEDE_EXISTING"

        # if both are high-confidence but not safe to auto-supersede -> conflict
        if candidate_conf >= HIGH_CONFIDENCE_THRESHOLD and existing_conf >= HIGH_CONFIDENCE_THRESHOLD:
            return "FLAG_CONFLICT"

        # otherwise unresolved difference -> uncertain
        return "MARK_UNCERTAIN"

    # 4) Same identity but not exact value, weaker/incomplete existing -> merge or supersede
    if same_identity(match) and not match.get("exact_identity_and_value"):
        if candidate_conf >= existing_conf and can_auto_supersede(candidate, existing):
            return "SUPERSEDE_EXISTING"
        return "MERGE_INTO_EXISTING"

    # 5) Weak relatedness -> merge
    if match.get("weak_related"):
        return "MERGE_INTO_EXISTING"

    # 6) Candidate explicitly uncertain should not auto-append as strong current truth
    if candidate_status == "uncertain":
        return "MARK_UNCERTAIN"

    # 7) Otherwise new fact
    return "APPEND_NEW"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--existing", required=True)
    args = parser.parse_args()

    candidate = load_json(args.candidate)
    existing = load_json(args.existing)

    result = subprocess.run(
        [
            "python3",
            str(MATCH_SCRIPT),
            "--candidate", args.candidate,
            "--existing", args.existing,
        ],
        capture_output=True,
        text=True,
        check=True
    )

    match = json.loads(result.stdout)
    outcome = classify_outcome(candidate, existing, match)

    out = {
        "outcome": outcome,
        "candidate_id": candidate.get("id"),
        "existing_id": existing.get("id"),
        "candidate_status": normalized_status(candidate),
        "existing_status": normalized_status(existing),
        "candidate_confidence": confidence_of(candidate),
        "existing_confidence": confidence_of(existing),
        "match": match
    }

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
