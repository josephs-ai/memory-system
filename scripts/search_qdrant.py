"""
Search and retrieve from qdrant.

Key functions: main
"""
import argparse
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from vector_store_qdrant import search_memory_vectors

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    query_vec = model.encode(
        [args.query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]

    results = search_memory_vectors(query_vec, limit=args.limit)

    if not results:
        print("NONE")
        return

    for r in results:
        payload = r.payload or {}
        print(
            {
                "score": r.score,
                "id": payload.get("id"),
                "text": payload.get("text"),
                "entity": payload.get("entity"),
                "property": payload.get("property"),
                "value": payload.get("value"),
                "scope": payload.get("scope"),
            }
        )


if __name__ == "__main__":
    main()
