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


# Payload field used for temporal window filtering (since/until/as_of). Stored
# as epoch SECONDS (integer) so Qdrant Range filters work cleanly and avoid
# timezone/string-compare pitfalls. Resolution order mirrors temporal.py:
#   effective_ts = COALESCE(last_confirmed, first_seen, created_at)
EFFECTIVE_TS_FIELD = "effective_ts_epoch"


def effective_ts_epoch(item: dict) -> int | None:
    """Resolve the effective timestamp of a memory item to epoch SECONDS (int).

    Mirrors temporal.resolve_effective_ts COALESCE order
    (last_confirmed > first_seen > created_at). Accepts datetime or ISO/str.
    Returns None if no usable timestamp — caller should OMIT the field so the
    point is treated as undated (kept by windows, matching SQL semantics).
    """
    from datetime import datetime, timezone

    for key in ("last_confirmed", "first_seen", "created_at"):
        v = item.get(key)
        if v is None:
            continue
        dt = None
        if isinstance(v, datetime):
            dt = v if v.tzinfo else v.replace(tzinfo=timezone.utc)
        else:
            s = str(v).strip()
            if not s:
                continue
            s = s.replace("T", " ")
            for cut in ("+", "Z"):
                if cut in s:
                    s = s.split(cut)[0]
            s = s.strip()
            for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try:
                    dt = datetime.strptime(s[:26], fmt).replace(tzinfo=timezone.utc)
                    break
                except ValueError:
                    continue
        if dt is not None:
            return int(dt.timestamp())
    return None


def ensure_qdrant_collection() -> None:
    client = get_qdrant_client()
    collections = client.get_collections().collections
    names = {c.name for c in collections}
    if QDRANT_COLLECTION not in names:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(
                size=QDRANT_VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )

    # Idempotent: ensure a payload index on the temporal field so Range filters
    # are efficient. create_payload_index is a no-op if the index already exists
    # (Qdrant returns ok); wrap defensively so collection setup never fails on it.
    try:
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name=EFFECTIVE_TS_FIELD,
            field_schema=models.PayloadSchemaType.INTEGER,
        )
    except Exception:
        pass

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

def build_temporal_filter(
    since_epoch: int | None = None,
    until_epoch: int | None = None,
) -> "models.Filter | None":
    """Build a Qdrant filter for the temporal window on EFFECTIVE_TS_FIELD.

    Semantics match the SQL push-down: a point is kept if its effective ts is
    within [since, until] OR it has no effective ts at all (undated → kept).
    We express "in range OR undated" as a should-clause (logical OR). When no
    bounds are given, returns None (no filtering).

    `as_of` is folded into `until_epoch` by the caller (as_of = upper bound).
    """
    if since_epoch is None and until_epoch is None:
        return None

    rng = models.Range(
        gte=float(since_epoch) if since_epoch is not None else None,
        lte=float(until_epoch) if until_epoch is not None else None,
    )
    in_window = models.FieldCondition(key=EFFECTIVE_TS_FIELD, range=rng)
    # Undated points (field absent) are kept, mirroring SQL "(EFF IS NULL OR ...)".
    undated = models.IsEmptyCondition(is_empty=models.PayloadField(key=EFFECTIVE_TS_FIELD))
    return models.Filter(should=[in_window, undated])


def search_memory_vectors(
    query_vector: Sequence[float],
    limit: int = 10,
    since_epoch: int | None = None,
    until_epoch: int | None = None,
) -> list:
    client = get_qdrant_client()
    query_filter = build_temporal_filter(since_epoch, until_epoch)
    result = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=list(query_vector),
        limit=limit,
        with_payload=True,
        query_filter=query_filter,
    )
    return result.points
