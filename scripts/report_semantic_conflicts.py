"""
Generate report on semantic conflicts.

Key functions: preview, looks_conflicting, fetch_candidate_pairs, main
"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from memory_db import close_pool, get_conn


OPPOSITES = [
    ("use ", "avoid "),
    ("requires ", "does not require "),
    ("enabled", "disabled"),
    ("available", "unavailable"),
    ("supported", "not supported"),
    ("should", "should not"),
    ("must", "must not"),
]


def preview(text, n=180):
    s = (text or "").replace("\n", " ").strip()
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


def looks_conflicting(a_text: str, b_text: str) -> bool:
    a_text = (a_text or "").lower()
    b_text = (b_text or "").lower()

    for x, y in OPPOSITES:
        if (x in a_text and y in b_text) or (y in a_text and x in b_text):
            return True

    return False


def fetch_candidate_pairs(limit: int):
    pair_clauses = []
    params = []

    for x, y in OPPOSITES:
        pair_clauses.append(
            """
            (
                (LOWER(COALESCE(a.text, '')) LIKE %s AND LOWER(COALESCE(b.text, '')) LIKE %s)
                OR
                (LOWER(COALESCE(a.text, '')) LIKE %s AND LOWER(COALESCE(b.text, '')) LIKE %s)
            )
            """
        )
        params.extend([f"%{x}%", f"%{y}%", f"%{y}%", f"%{x}%"])

    where_opposites = " OR ".join(pair_clauses)

    sql = f"""
        SELECT
            a.id,
            a.entity,
            a.property,
            a.value,
            a.text,
            b.id,
            b.entity,
            b.property,
            b.value,
            b.text
        FROM memory_items a
        JOIN memory_items b
          ON a.entity = b.entity
         AND a.id < b.id
        WHERE a.status = 'active'
          AND b.status = 'active'
          AND a.entity IS NOT NULL
          AND a.entity <> 'general'
          AND ({where_opposites})
        ORDER BY a.entity, a.id, b.id
        LIMIT %s
    """
    params.append(limit)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def main():
    limit = 200
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
        except Exception:
            pass

    rows = fetch_candidate_pairs(limit)

    print("=== SEMANTIC CONFLICT CANDIDATES ===")
    print()

    found = False
    for row in rows:
        (
            a_id,
            a_entity,
            a_property,
            a_value,
            a_text,
            b_id,
            b_entity,
            b_property,
            b_value,
            b_text,
        ) = row

        if not looks_conflicting(a_text, b_text):
            continue

        found = True
        print(f"id1={a_id}")
        print(f"entity1={a_entity} property1={a_property} value1={a_value}")
        print(f"text1={preview(a_text)}")
        print(f"id2={b_id}")
        print(f"entity2={b_entity} property2={b_property} value2={b_value}")
        print(f"text2={preview(b_text)}")
        print()

    if not found:
        print("NONE")

    close_pool()


if __name__ == "__main__":
    main()
