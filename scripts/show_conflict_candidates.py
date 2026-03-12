import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from memory_db import close_pool, fetch_memory_items


def same_slot(a: dict, b: dict) -> bool:
    return (
        a.get("memory_type") == b.get("memory_type")
        and a.get("entity") == b.get("entity")
        and a.get("property") == b.get("property")
        and a.get("scope") == b.get("scope")
    )


def preview(text, n=160):
    s = (text or "").replace("\n", " ").strip()
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


def main():
    active = fetch_memory_items(["active"])

    print("=== CONFLICT CANDIDATES ===")
    print()

    found = False
    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            a = active[i]
            b = active[j]

            if same_slot(a, b) and a.get("value") != b.get("value"):
                found = True
                print(f"A={a.get('id')} B={b.get('id')}")
                print(
                    f"slot memory_type={a.get('memory_type')} "
                    f"entity={a.get('entity')} property={a.get('property')} scope={a.get('scope')}"
                )
                print(f"A value={a.get('value')} text={preview(a.get('text'))}")
                print(f"B value={b.get('value')} text={preview(b.get('text'))}")
                print()

    if not found:
        print("NONE")

    close_pool()


if __name__ == "__main__":
    main()
