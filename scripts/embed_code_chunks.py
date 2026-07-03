"""
embed_code_chunks.py — Embed code chunks into Qdrant for vector retrieval.

Slice 3 (S3) of the Zero-Reread Architecture.

Creates and populates a dedicated `code_chunks` Qdrant collection,
completely isolated from the existing `memory_items` collection.
Embeds the concatenation of (qualified_name + signature + docstring + body)
for each chunk, producing rich semantic vectors for code search.

Usage:
    python embed_code_chunks.py [--batch-size 64] [--device cpu] [--force]

Features:
    - Dedicated `code_chunks` Qdrant collection (isolated from memory_items)
    - Incremental: only embeds chunks not yet in Qdrant (unless --force)
    - Batched embedding for efficiency
    - Batched Qdrant upserts (reduces API calls)
    - Dual-write: PostgreSQL code_chunk_embeddings + Qdrant
    - Rich payload for filtered search (chunk_type, module_name, project_id, etc.)
"""

from __future__ import annotations

import argparse
import logging
import os
import uuid
from typing import Any

import psycopg
import torch
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

LOGGER = logging.getLogger("openclaw.embed_code_chunks")
logging.basicConfig(
    level=os.environ.get("OPENCLAW_AST_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

DB_DSN = (os.environ.get("OPENCLAW_MEMORY_DSN") or os.environ.get("OPENCLAW_MEMORY_DB_DSN") or "dbname=openclaw_memory")  # honor both canonical DSN env names
QDRANT_HOST = os.environ.get("OPENCLAW_QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("OPENCLAW_QDRANT_PORT", "6333"))
COLLECTION_NAME = "code_chunks"  # Isolated from "memory_items"
VECTOR_SIZE = 384  # all-MiniLM-L6-v2 dimension
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Qdrant helpers (isolated collection)
# ---------------------------------------------------------------------------

def get_qdrant_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def ensure_code_chunks_collection() -> None:
    """Create the code_chunks Qdrant collection if it doesn't exist."""
    client = get_qdrant_client()
    collections = client.get_collections().collections
    names = {c.name for c in collections}
    if COLLECTION_NAME in names:
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE,
        ),
    )
    LOGGER.info("Created Qdrant collection: %s", COLLECTION_NAME)


def code_chunk_point_id(chunk_id: str) -> str:
    """Deterministic Qdrant point ID from chunk ID."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"openclaw-code-chunk:{chunk_id}"))


def upsert_code_vectors_batch(
    client: QdrantClient,
    points: list[models.PointStruct],
) -> None:
    """Batch upsert points into the code_chunks collection."""
    if not points:
        return
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )


# ---------------------------------------------------------------------------
# PostgreSQL embedding table
# ---------------------------------------------------------------------------

def ensure_embedding_table(conn) -> None:
    """Create the code_chunk_embeddings table if it doesn't exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS code_chunk_embeddings (
                chunk_id    TEXT PRIMARY KEY REFERENCES code_chunks(id) ON DELETE CASCADE,
                model_name  TEXT NOT NULL,
                embedding   vector(384) NOT NULL,
                updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_code_chunk_embeddings_cosine
            ON code_chunk_embeddings USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 50)
        """)
    conn.commit()


def upsert_pg_embedding(conn, chunk_id: str, model_name: str, embedding: list[float]) -> None:
    """Upsert a single embedding into PostgreSQL."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO code_chunk_embeddings (chunk_id, model_name, embedding, updated_at)
            VALUES (%s, %s, %s, now())
            ON CONFLICT (chunk_id) DO UPDATE SET
                model_name = EXCLUDED.model_name,
                embedding = EXCLUDED.embedding,
                updated_at = now()
            """,
            (chunk_id, model_name, embedding),
        )
    # Don't commit per-row; caller commits per batch


# ---------------------------------------------------------------------------
# Fetch chunks needing embedding
# ---------------------------------------------------------------------------

def fetch_chunks_to_embed(conn, *, force: bool = False) -> list[dict[str, Any]]:
    """
    Fetch code chunks that need embedding.
    If force=False, only fetch chunks not yet in code_chunk_embeddings.
    """
    with conn.cursor() as cur:
        if force:
            cur.execute("""
                SELECT id, file_path, chunk_type, qualified_name, module_name,
                       signature, docstring, body, project_id, subproject_id,
                       line_start, line_end, class_name
                FROM code_chunks
                ORDER BY id
            """)
        else:
            cur.execute("""
                SELECT c.id, c.file_path, c.chunk_type, c.qualified_name, c.module_name,
                       c.signature, c.docstring, c.body, c.project_id, c.subproject_id,
                       c.line_start, c.line_end, c.class_name
                FROM code_chunks c
                LEFT JOIN code_chunk_embeddings e ON e.chunk_id = c.id
                WHERE e.chunk_id IS NULL
                ORDER BY c.id
            """)
        cols = [desc[0] for desc in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# Text construction for embedding
# ---------------------------------------------------------------------------

def build_embedding_text(chunk: dict[str, Any]) -> str:
    """
    Build the text to embed for a code chunk.
    Concatenates: qualified_name + signature + docstring + body
    This gives the vector model both semantic meaning and structural context.
    """
    parts = []

    # Qualified name provides search by name
    if chunk.get("qualified_name"):
        parts.append(chunk["qualified_name"])

    # Signature provides parameter/type information
    if chunk.get("signature"):
        parts.append(chunk["signature"])

    # Docstring provides natural language description
    if chunk.get("docstring"):
        parts.append(chunk["docstring"])

    # Body provides full implementation context
    # Truncate very large bodies to keep embedding meaningful
    body = chunk.get("body") or ""
    if len(body) > 4000:
        body = body[:4000] + "\n# ... truncated"
    parts.append(body)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def choose_device(force_device: str | None = None) -> str:
    if force_device:
        return force_device
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def chunked(items: list, size: int):
    """Yield successive chunks from a list."""
    for i in range(0, len(items), size):
        yield items[i:i + size]


def main():
    parser = argparse.ArgumentParser(
        description="Embed code chunks into Qdrant for vector retrieval."
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--force", action="store_true",
                        help="Re-embed all chunks, not just new ones")
    args = parser.parse_args()

    device = choose_device(args.device)
    LOGGER.info("Loading embedding model: %s (device=%s)", MODEL_NAME, device)
    model = SentenceTransformer(MODEL_NAME, device=device)

    conn = psycopg.connect(DB_DSN)
    qdrant = get_qdrant_client()

    try:
        # Ensure infrastructure exists
        ensure_code_chunks_collection()
        ensure_embedding_table(conn)

        # Fetch chunks needing embedding
        chunks = fetch_chunks_to_embed(conn, force=args.force)
        if not chunks:
            print("NONE")
            return

        total = len(chunks)
        embedded = 0
        LOGGER.info("Embedding %d code chunks", total)

        for batch in chunked(chunks, args.batch_size):
            # Build texts for this batch
            texts = [build_embedding_text(c) for c in batch]

            # Encode batch
            embeddings = model.encode(
                texts,
                batch_size=args.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            # Prepare Qdrant points batch
            qdrant_points = []
            # strict=True: chunks and embeddings must stay 1:1 (written to PG +
            # Qdrant keyed by chunk["id"]); a count mismatch must fail loud
            # rather than silently corrupt the code index.
            for chunk, emb in zip(batch, embeddings, strict=True):
                vector = emb.tolist()

                # Write to PostgreSQL
                upsert_pg_embedding(conn, chunk["id"], MODEL_NAME, vector)

                # Prepare Qdrant point
                qdrant_points.append(
                    models.PointStruct(
                        id=code_chunk_point_id(chunk["id"]),
                        vector=vector,
                        payload={
                            "chunk_id": chunk["id"],
                            "file_path": chunk["file_path"],
                            "chunk_type": chunk["chunk_type"],
                            "qualified_name": chunk["qualified_name"],
                            "module_name": chunk["module_name"],
                            "signature": chunk["signature"],
                            "docstring": chunk["docstring"],
                            "project_id": chunk["project_id"],
                            "line_start": chunk["line_start"],
                            "line_end": chunk["line_end"],
                            "class_name": chunk["class_name"],
                            # Body intentionally excluded from payload to save space;
                            # it's in PostgreSQL and can be fetched by chunk_id
                        },
                    )
                )

            # Batch upsert to Qdrant
            upsert_code_vectors_batch(qdrant, qdrant_points)

            # Commit PostgreSQL batch
            conn.commit()

            embedded += len(batch)
            print(f"PROGRESS {embedded}/{total}")

        print(f"EMBEDDED={embedded}")
        print(f"MODEL={MODEL_NAME}")
        print(f"DEVICE={device}")
        print(f"COLLECTION={COLLECTION_NAME}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
