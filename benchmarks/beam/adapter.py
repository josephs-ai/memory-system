"""
beam/adapter.py — BEAM benchmark dataset adapter.

Dataset: Mohammadta/BEAM (HuggingFace)
BEAM tests 10 memory ability types across multi-session conversations:
  - abstention, contradiction_resolution, event_ordering,
  - information_extraction, instruction_following, knowledge_update,
  - multi_session_reasoning, preference_following, summarization,
  - temporal_reasoning

Sizes: 100K, 500K, 1M, 10M tokens per conversation set.
Each conversation has 20 probing questions (2 per type).
"""

from __future__ import annotations

import ast
import json
import logging
import os
from pathlib import Path
from typing import Any

BENCHMARKS_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BENCHMARKS_DIR / "datasets" / "beam"
LOGGER = logging.getLogger("openclaw.benchmarks.beam")

QUESTION_TYPES = [
    "abstention",
    "contradiction_resolution",
    "event_ordering",
    "information_extraction",
    "instruction_following",
    "knowledge_update",
    "multi_session_reasoning",
    "preference_following",
    "summarization",
    "temporal_reasoning",
]


def download_dataset(chat_size: str = "1M") -> list[dict]:
    """Download BEAM dataset from HuggingFace and cache locally."""
    os.makedirs(DATASETS_DIR, exist_ok=True)
    cache_path = DATASETS_DIR / f"beam_{chat_size}.json"

    if cache_path.exists():
        LOGGER.info("Loading cached %s dataset: %s", chat_size, cache_path)
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    LOGGER.info("Downloading BEAM %s dataset from HuggingFace...", chat_size)
    from datasets import load_dataset as hf_load

    if chat_size == "10M":
        ds = hf_load("Mohammadta/BEAM-10M", split="10M")
    else:
        ds = hf_load("Mohammadta/BEAM", split=chat_size)

    conversations = []
    for idx, item in enumerate(ds):
        conv: dict[str, Any] = {
            "conversation_id": item.get("conversation_id", f"{chat_size}_{idx}"),
            "chat": item.get("chat", []),
            "user_profile": item.get("user_profile", {}),
        }

        # Parse probing_questions (may be stored as string repr)
        pq_raw = item.get("probing_questions", "{}")
        if isinstance(pq_raw, str):
            try:
                conv["probing_questions"] = ast.literal_eval(pq_raw)
            except (ValueError, SyntaxError):
                try:
                    conv["probing_questions"] = json.loads(pq_raw)
                except json.JSONDecodeError:
                    conv["probing_questions"] = {}
        else:
            conv["probing_questions"] = pq_raw if isinstance(pq_raw, dict) else {}

        conversations.append(conv)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False)
    LOGGER.info("Cached %d conversations to %s", len(conversations), cache_path)

    return conversations


def parse_chat_to_turns(chat_data: Any) -> list[dict]:
    """Parse BEAM chat data into a flat list of turn dicts with role+content."""
    if not chat_data:
        return []

    turns = []

    # 2D list format (1M and smaller)
    if isinstance(chat_data, list):
        if chat_data and isinstance(chat_data[0], list):
            for batch in chat_data:
                for turn in batch:
                    if isinstance(turn, dict) and turn.get("content"):
                        turns.append({
                            "role": turn.get("role", "user"),
                            "content": turn["content"],
                            "time_anchor": turn.get("time_anchor"),
                        })
        # Flat list of dicts
        elif chat_data and isinstance(chat_data[0], dict):
            if "turns" in chat_data[0]:
                # Batch-dict format
                for batch in chat_data:
                    for turn in batch.get("turns", []):
                        if isinstance(turn, list):
                            for t in turn:
                                if isinstance(t, dict) and t.get("content"):
                                    turns.append({
                                        "role": t.get("role", "user"),
                                        "content": t["content"],
                                        "time_anchor": t.get("time_anchor"),
                                    })
                        elif isinstance(turn, dict) and turn.get("content"):
                            turns.append({
                                "role": turn.get("role", "user"),
                                "content": turn["content"],
                                "time_anchor": turn.get("time_anchor"),
                            })
            elif "role" in chat_data[0] or "content" in chat_data[0]:
                for turn in chat_data:
                    if isinstance(turn, dict) and turn.get("content"):
                        turns.append({
                            "role": turn.get("role", "user"),
                            "content": turn["content"],
                            "time_anchor": turn.get("time_anchor"),
                        })

    return turns


def extract_probing_questions(conversation: dict) -> list[dict]:
    """Extract all probing questions from a BEAM conversation."""
    pq = conversation.get("probing_questions", {})
    if not pq or not isinstance(pq, dict):
        return []

    questions = []
    for q_type in QUESTION_TYPES:
        type_questions = pq.get(q_type, [])
        if isinstance(type_questions, list):
            for q in type_questions:
                if isinstance(q, dict):
                    q["question_type"] = q_type
                    questions.append(q)
        elif isinstance(type_questions, dict):
            type_questions["question_type"] = q_type
            questions.append(type_questions)

    return questions


def extract_rubric_nuggets(question_data: dict) -> list[str]:
    """Extract rubric nugget descriptions from a question dict."""
    rubric_raw = question_data.get("rubric", {})
    if isinstance(rubric_raw, dict):
        nuggets = rubric_raw.get("nuggets", [])
        return [
            n.get("description", str(n)) if isinstance(n, dict) else str(n)
            for n in nuggets
        ]
    if isinstance(rubric_raw, list):
        return [str(n) for n in rubric_raw]
    if rubric_raw:
        return [str(rubric_raw)]
    return []


def chat_turns_to_memory_items(
    turns: list[dict],
    *,
    source_agent: str,
    source_session: str,
    conversation_id: str,
) -> list[dict]:
    """Convert chat turns into memory items for ingestion.

    Each turn becomes a memory item. We extract sentences from longer
    turns to improve retrieval granularity.
    """
    import sys
    sys.path.insert(0, str(BENCHMARKS_DIR))
    from common import make_memory_item

    items = []
    for i, turn in enumerate(turns):
        content = turn["content"].strip()
        role = turn.get("role", "user")
        if not content:
            continue

        # For long turns, split into sentences for finer-grained retrieval
        sentences = _split_sentences(content)
        if len(sentences) > 1 and len(content) > 200:
            for j, sent in enumerate(sentences):
                if len(sent.strip()) < 15:
                    continue
                item = make_memory_item(
                    f"{role}: {sent}",
                    source_agent=source_agent,
                    source_session=source_session,
                    entity=f"conv_{conversation_id}",
                    property="utterance",
                    value=sent[:200],
                    memory_type="fact",
                    scope="benchmark",
                    tags=["beam", f"turn_{i}", f"sent_{j}", role],
                )
                if turn.get("time_anchor"):
                    item["first_seen"] = turn["time_anchor"]
                    item["last_confirmed"] = turn["time_anchor"]
                items.append(item)
        else:
            item = make_memory_item(
                f"{role}: {content}",
                source_agent=source_agent,
                source_session=source_session,
                entity=f"conv_{conversation_id}",
                property="utterance",
                value=content[:200],
                memory_type="fact",
                scope="benchmark",
                tags=["beam", f"turn_{i}", role],
            )
            if turn.get("time_anchor"):
                item["first_seen"] = turn["time_anchor"]
                item["last_confirmed"] = turn["time_anchor"]
            items.append(item)

    return items


def _split_sentences(text: str) -> list[str]:
    """Simple sentence splitter."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s for s in sentences if s.strip()]
