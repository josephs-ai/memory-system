"""
longmemeval/adapter.py — LongMemEval dataset adapter.

Dataset: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
Files:
  - longmemeval_oracle.json   (oracle context)
  - longmemeval_s_cleaned.json (short context, 500 questions)
  - longmemeval_m_cleaned.json (medium context, 500 questions)

Each entry:
  {
    "question_id": str,
    "question_type": str,     # information_extraction | multi_session | knowledge_update |
                               # temporal_reasoning | abstention
    "question": str,
    "answer": str,
    "sessions": [             # list of conversation sessions
      [                       # session = list of turns
        {"role": "user"|"assistant", "content": str},
        ...
      ]
    ]
  }

Eval approach:
  1. Load dataset subset
  2. For each entry: ingest sessions into isolated namespace
  3. Query with question text
  4. Score retrieved context (exact-match F1 + LLM judge)
  5. Cleanup
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Iterator

BENCHMARKS_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BENCHMARKS_DIR / "datasets" / "longmemeval"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

LOGGER = logging.getLogger("openclaw.benchmarks.longmemeval")

# Dataset file URLs (raw HuggingFace dataset files)
DATASET_FILES = {
    "oracle": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json",
    "s": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json",
    "m": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_m_cleaned.json",
}

LOCAL_PATHS = {
    "oracle": DATASETS_DIR / "longmemeval_oracle.json",
    "s": DATASETS_DIR / "longmemeval_s_cleaned.json",
    "m": DATASETS_DIR / "longmemeval_m_cleaned.json",
}


def download_dataset(split: str = "s", force: bool = False) -> Path:
    """
    Download LongMemEval dataset file if not already present.
    split: "oracle" | "s" | "m"
    """
    local_path = LOCAL_PATHS[split]
    if local_path.exists() and not force:
        LOGGER.info("Dataset already exists: %s", local_path)
        return local_path

    url = DATASET_FILES[split]
    LOGGER.info("Downloading %s → %s", url, local_path)

    try:
        import httpx
        with httpx.Client(follow_redirects=True, timeout=120) as client:
            response = client.get(url)
            response.raise_for_status()
            local_path.write_bytes(response.content)
        LOGGER.info("Downloaded %d bytes", local_path.stat().st_size)
    except Exception as e:
        LOGGER.error("Download failed: %s", e)
        raise

    return local_path


def load_dataset(split: str = "s", max_entries: int | None = None) -> list[dict]:
    """Load and return LongMemEval entries."""
    local_path = LOCAL_PATHS[split]
    if not local_path.exists():
        download_dataset(split)

    with open(local_path, encoding="utf-8") as f:
        data = json.load(f)

    if max_entries is not None:
        data = data[:max_entries]

    LOGGER.info("Loaded %d entries from %s split", len(data), split)
    return data


def sessions_to_memory_items(
    sessions: list[list[dict]],
    *,
    source_agent: str,
    source_session: str,
) -> list[dict]:
    """
    Convert chat sessions to memory items.

    Each turn becomes a memory item:
      text = "<role>: <content>"
      entity = role
      property = "utterance"
      value = content[:200]  # truncated
    """
    import sys
    sys.path.insert(0, str(BENCHMARKS_DIR))
    from common import make_memory_item

    items = []
    for session_idx, session in enumerate(sessions):
        for turn_idx, turn in enumerate(session):
            role = turn.get("role", "unknown")
            content = turn.get("content", "").strip()
            if not content:
                continue

            text = f"{role}: {content}"
            item = make_memory_item(
                text,
                source_agent=source_agent,
                source_session=source_session,
                entity=role,
                property="utterance",
                value=content[:200],
                memory_type="conversation",
                scope="benchmark",
                tags=["longmemeval", f"session_{session_idx}"],
                item_id=f"{source_session}_s{session_idx}_t{turn_idx}",
            )
            items.append(item)

    return items


def extract_answer_from_context(
    question: str,
    retrieved_items: list[dict],
    gold_answer: str,
) -> str:
    """
    Simple extraction: return the most relevant snippet from retrieved context.
    For benchmarking, we use a heuristic: return the value field or first 200 chars.
    """
    if not retrieved_items:
        return ""

    # Use value field if available (structured memory), else use text
    for item in retrieved_items:
        val = item.get("value", "")
        if val and len(val) > 5:
            return val

    # Fall back to top item text
    top_text = retrieved_items[0].get("text", "")
    # Extract assistant turn if present
    if "assistant:" in top_text.lower():
        parts = top_text.split(":", 1)
        if len(parts) > 1:
            return parts[1].strip()[:300]

    return top_text[:300]
