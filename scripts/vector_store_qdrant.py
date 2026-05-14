"""
Qdrant vector store adapter for storing and searching embeddings.
Handles collection management, upsert, and similarity search.
"""
from __future__ import annotations

import os
from typing import Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models

import uuid

QDRANT_HOST = os.environ.get("OPENCLAW_QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("OPENCLAW_QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.environ.get("OPENCLAW_QDRANT_COLLECTION", "memory_items")
QDRANT_VECTOR_SIZE = int(os.environ.get("OPENCLAW_QDRANT_VECTOR_SIZE", "384"))


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def ensure_qdrant_collection() -> None:
    client = get_qdrant_client()
    collections = client.get_collections().collections
    names = {c.name for c in collections}
    if QDRANT_COLLECTION in names:
        return

    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=models.VectorParams(
            size=QDRANT_VECTOR_SIZE,
            distance=models.Distance.COSINE,
        ),
    )

def qdrant_point_id(memory_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"openclaw-memory:{memory_id}"))

def upsert_memory_vector(
    memory_id: str,
    vector: Sequence[float],
    payload: dict,
) -> None:
    client = get_qdrant_client()
    client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=[
            models.PointStruct(
                id=qdrant_point_id(memory_id),
                vector=list(vector),
                payload=payload,
            )
        ],
    )

def search_memory_vectors(
    query_vector: Sequence[float],
    limit: int = 10,
) -> list:
    client = get_qdrant_client()
    result = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=list(query_vector),
        limit=limit,
        with_payload=True,
    )
    return result.points
