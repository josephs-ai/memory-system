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
from vector_store_qdrant import (
    EFFECTIVE_TS_FIELD,
    effective_ts_epoch,
    ensure_qdrant_collection,
    upsert_memory_vector,
)

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
    args = parser.parse_args()

    items = fetch_memory_items_for_embedding(args.status, MODEL_NAME)

    if not items:
        print("NONE")
        close_pool()
        return

    device = choose_device(args.device)
    model = SentenceTransformer(MODEL_NAME, device=device)
    ensure_qdrant_collection()

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

        # strict=True: items and their embeddings MUST stay 1:1. A silent
        # length mismatch (encoder returning fewer/more vectors than inputs)
        # would otherwise mis-pair or drop embeddings, corrupting the index
        # keyed by item["id"]. Fail loud instead.
        for item, emb in zip(batch, embeddings, strict=True):
            vector = emb.tolist()

            upsert_memory_embedding(item["id"], MODEL_NAME, vector)

            payload = {
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
            }
            # Temporal filtering support: store the effective ts as epoch seconds
            # (omit when undated so the point is treated as undated by windows).
            _eff = effective_ts_epoch(item)
            if _eff is not None:
                payload[EFFECTIVE_TS_FIELD] = _eff

            upsert_memory_vector(
                memory_id=item["id"],
                vector=vector,
                payload=payload,
            )

            embedded += 1

        print(f"PROGRESS {embedded}/{total}")

    print(f"EMBEDDED={embedded}")
    print(f"MODEL={MODEL_NAME}")
    print(f"DEVICE={device}")
    close_pool()


if __name__ == "__main__":
    main()
