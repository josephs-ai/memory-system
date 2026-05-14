"""
Memory system utility: check agent memory policy.

Key functions: load_json, main
"""
import json
import argparse
from pathlib import Path

WORKSPACE = Path.home() / ".openclaw" / "workspace"
REVIEW_DIR = WORKSPACE / "memory" / "review"
POLICY_FILE = REVIEW_DIR / "agent-memory-policy.json"

def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True)
    parser.add_argument("--item", required=True)
    args = parser.parse_args()

    policy = load_json(POLICY_FILE)
    item = load_json(Path(args.item))

    agent_policy = policy.get(args.agent, policy["default"])

    memory_type = item.get("memory_type")
    scope = item.get("scope")
    confidence = float(item.get("confidence", 0.0))

    allowed_types = agent_policy["allowed_memory_types"]
    allowed_scopes = agent_policy["allowed_scopes"]
    min_auto = float(agent_policy["min_confidence_auto"])
    min_inbox = float(agent_policy["min_confidence_inbox"])

    if memory_type not in allowed_types:
        print("DISCARD type_not_allowed")
        return

    if scope not in allowed_scopes:
        print("DISCARD scope_not_allowed")
        return

    explicit_authority = str(item.get("authority_basis") or "").strip().lower() in {"user_explicit", "developer_explicit", "tester_explicit", "system_tool_verified"}
    explicit_approval = bool(item.get("approved_by") or item.get("approval_source"))
    if confidence >= min_auto and (explicit_authority and (explicit_approval or str(item.get("authority_basis") or "").strip().lower() == "system_tool_verified")):
        print("AUTO")
        return

    if confidence >= min_inbox:
        print("INBOX")
        return

    print("DISCARD below_threshold")

if __name__ == "__main__":
    main()
