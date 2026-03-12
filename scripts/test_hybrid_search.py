import argparse
import json
import sys
from pathlib import Path
from decimal import Decimal

from sentence_transformers import SentenceTransformer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from memory_db import hybrid_search_memory_items, close_pool

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def json_safe(value):
    if isinstance(value, Decimal):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    model = SentenceTransformer(MODEL_NAME)
    query_embedding = model.encode(
        args.query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    rows = hybrid_search_memory_items(
        query_embedding,
        query_text=args.query,
        status="active",
        allowed_sensitivities=["public", "internal"],
        limit=args.top_k,
    )

    if not rows:
        print("NONE")
        close_pool()
        return

    for row in rows:
        print(json.dumps({
            "id": row.get("id"),
            "final_score": row.get("final_score"),
            "score_fts": row.get("fts_rank"),
            "score_vector": row.get("vector_score"),
            "structured_bonus": row.get("structured_bonus"),
            "importance_bonus": row.get("importance_bonus"),
            "text": row.get("text"),
            "entity": row.get("entity"),
            "property": row.get("property"),
            "value": row.get("value"),
            "sensitivity": row.get("sensitivity"),
        }, ensure_ascii=False, default=json_safe))

    close_pool()


if __name__ == "__main__":
    main()
