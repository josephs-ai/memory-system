"""
Memory system utility: match memory items.

Key functions: load_json, norm, same, present
"""
import json
import argparse
from pathlib import Path

def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def norm(x):
    if x is None:
        return None
    if isinstance(x, str):
        s = x.strip()
        return s if s else None
    return x

def same(a, b, field):
    return norm(a.get(field)) == norm(b.get(field))

def present(x, field):
    return norm(x.get(field)) is not None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--existing", required=True)
    args = parser.parse_args()

    candidate = load_json(args.candidate)
    existing = load_json(args.existing)

    same_type = same(candidate, existing, "memory_type")
    same_entity = same(candidate, existing, "entity")
    same_property = same(candidate, existing, "property")
    same_scope = same(candidate, existing, "scope")
    same_value = same(candidate, existing, "value")

    comparable_identity = (
        same_type and
        same_entity and
        same_scope and
        (
            same_property or
            (not present(candidate, "property") and not present(existing, "property"))
        )
    )

    exact_identity_and_value = comparable_identity and same_value

    same_slot_different_value = (
        same_type and
        same_entity and
        same_scope and
        same_property and
        present(candidate, "value") and
        present(existing, "value") and
        not same_value
    )

    possible_value_conflict = same_slot_different_value

    weak_related = (
        same_type and
        same_entity and
        same_scope and
        not comparable_identity
    )

    result = {
        "same_type": same_type,
        "same_entity": same_entity,
        "same_property": same_property,
        "same_scope": same_scope,
        "same_value": same_value,
        "comparable_identity": comparable_identity,
        "exact_identity_and_value": exact_identity_and_value,
        "same_slot_different_value": same_slot_different_value,
        "possible_value_conflict": possible_value_conflict,
        "weak_related": weak_related
    }

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
