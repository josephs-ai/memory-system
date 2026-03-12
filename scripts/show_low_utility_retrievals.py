import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from memory_db import low_utility_feedback_rows, close_pool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-bad", type=int, default=1)
    args = parser.parse_args()

    rows = low_utility_feedback_rows(args.min_bad)

    print("=== LOW UTILITY RETRIEVALS ===")
    print()

    if not rows:
        print("NONE")
        close_pool()
        return

    for row in rows:
        print(
            f"id={row['selected_id']} useful={row['useful']} bad={row['bad']}"
        )

    close_pool()


if __name__ == "__main__":
    main()
