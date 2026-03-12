import argparse
import sys
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from memory_db import fetch_memory_items_for_embedding, upsert_memory_embedding, close_pool

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

    device = choose_device(args.device)
    model = SentenceTransformer(MODEL_NAME, device=device)

    items = fetch_memory_items_for_embedding(args.status)

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
            upsert_memory_embedding(item["id"], MODEL_NAME, emb.tolist())
            embedded += 1

        print(f"PROGRESS {embedded}/{total}")

    print(f"EMBEDDED={embedded}")
    print(f"MODEL={MODEL_NAME}")
    print(f"DEVICE={device}")
    close_pool()


if __name__ == "__main__":
    main()
