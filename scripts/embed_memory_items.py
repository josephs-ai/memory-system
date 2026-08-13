"""
Generate sentence-transformer embeddings for memory items and store them
in Qdrant for vector similarity search.
"""
import argparse
import sys
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from memory_db import fetch_memory_items_for_embedding, upsert_memory_embedding, close_pool
from vector_store_qdrant import ensure_qdrant_collection, upsert_memory_vector

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def choose_device(force_device: str | None = None) -> str:
    if force_device:
        return force_device
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def chunked(items, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", default="active")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument(
        "--all",
        action="store_true",
        help="Re-embed every item, not just new/changed ones (e.g. after a model change).",
    )
    args = parser.parse_args()

    device = choose_device(args.device)
    model = SentenceTransformer(MODEL_NAME, device=device)
    ensure_qdrant_collection()

    items = fetch_memory_items_for_embedding(
        args.status, only_missing=not args.all, model_name=MODEL_NAME
    )

    if not items:
        print("NONE")
        close_pool()
        return

    total = len(items)
    embedded = 0

    for batch in chunked(items, args.batch_size):
        texts = [item["text"] or "" for item in batch]
        embeddings = model.encode(
            texts,
            batch_size=args.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        for item, emb in zip(batch, embeddings):
            vector = emb.tolist()

            upsert_memory_embedding(item["id"], MODEL_NAME, vector)

            upsert_memory_vector(
                memory_id=item["id"],
                vector=vector,
                payload={
                    "id": item["id"],
                    "text": item.get("text"),
                    "memory_type": item.get("memory_type"),
                    "scope": item.get("scope"),
                    "entity": item.get("entity"),
                    "property": item.get("property"),
                    "value": item.get("value"),
                    "status": item.get("status"),
                    "confidence": item.get("confidence"),
                    "importance": item.get("importance"),
                    "freshness_class": item.get("freshness_class"),
                    "source_agent": item.get("source_agent"),
                    "source_session": item.get("source_session"),
                    "source_chunk": item.get("source_chunk"),
                },
            )

            embedded += 1

        print(f"PROGRESS {embedded}/{total}")

    print(f"EMBEDDED={embedded}")
    print(f"MODEL={MODEL_NAME}")
    print(f"DEVICE={device}")
    close_pool()


if __name__ == "__main__":
    main()
