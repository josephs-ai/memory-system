"""
locomo/adapter.py — LoCoMo dataset adapter.

Dataset: https://github.com/snap-research/locomo → data/locomo10.json

Structure:
{
  "<conv_id>": {
    "conversation": [
      {
        "dia_id": int,
        "blended_skill_talk": str | null,
        "text": str,
        "speaker": "A" | "B",
        "date": str,
        "session": int,
        ...
      }
    ],
    "qa": [
      {
        "question": str,
        "answer": str,
        "category": str,       # single_hop | multi_hop | temporal | adversarial
        "evidences": [str],    # list of dialog text snippets
        "dia_id": [int] | null # evidence dialog IDs
      }
    ]
  }
}

10 very long conversations with QA annotations.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BENCHMARKS_DIR / "datasets" / "locomo"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

LOGGER = logging.getLogger("openclaw.benchmarks.locomo")

DATASET_URLS = [
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
    "https://github.com/snap-research/locomo/raw/main/data/locomo10.json",
    "https://huggingface.co/datasets/snap-research/locomo/resolve/main/data/locomo10.json",
]
LOCAL_PATH = DATASETS_DIR / "locomo10.json"


def download_dataset(force: bool = False) -> Path:
    """Download LoCoMo dataset if not present. Tries multiple mirrors."""
    if LOCAL_PATH.exists() and not force:
        LOGGER.info("Dataset already exists: %s", LOCAL_PATH)
        return LOCAL_PATH

    import httpx

    for url in DATASET_URLS:
        LOGGER.info("Trying LoCoMo download from %s", url)
        try:
            with httpx.Client(follow_redirects=True, timeout=180) as client:
                response = client.get(url)
                response.raise_for_status()
                LOCAL_PATH.write_bytes(response.content)
            LOGGER.info("Downloaded %d bytes", LOCAL_PATH.stat().st_size)
            return LOCAL_PATH
        except Exception as e:
            LOGGER.warning("URL %s failed: %s", url, e)
            continue

    raise RuntimeError(f"Failed to download LoCoMo dataset from all mirrors: {DATASET_URLS}")


def load_dataset(max_convs: int | None = None) -> dict:
    """Load LoCoMo dataset. Returns dict keyed by conversation ID."""
    if not LOCAL_PATH.exists():
        download_dataset()

    with open(LOCAL_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    # Normalize: dataset may be a list of dicts with 'sample_id'
    if isinstance(raw, list):
        data = {}
        for item in raw:
            cid = item.get("sample_id", str(len(data)))
            data[cid] = item
    else:
        data = raw

    if max_convs is not None:
        keys = list(data.keys())[:max_convs]
        data = {k: data[k] for k in keys}

    LOGGER.info("Loaded %d conversations", len(data))
    return data


def conversation_to_memory_items(
    conversation: dict | list,
    *,
    source_agent: str,
    source_session: str,
) -> list[dict]:
    """
    Convert LoCoMo conversation turns to memory items.
    Handles both flat list format and session-dict format.
    """
    import sys
    sys.path.insert(0, str(BENCHMARKS_DIR))
    from common import make_memory_item
    from fact_extractor import extract_typed_facts

    # Extract speaker names
    speaker_a = conversation.get("speaker_a", "A") if isinstance(conversation, dict) else "A"
    speaker_b = conversation.get("speaker_b", "B") if isinstance(conversation, dict) else "B"

    # Flatten sessions into a turn list
    turns = []
    if isinstance(conversation, dict):
        # Session-dict format: session_1, session_2, etc.
        session_idx = 1
        while True:
            sess_key = f"session_{session_idx}"
            date_key = f"session_{session_idx}_date_time"
            session_data = conversation.get(sess_key)
            if session_data is None:
                break
            date_str = conversation.get(date_key, "")
            if isinstance(session_data, list):
                for turn in session_data:
                    turn["_session"] = session_idx
                    turn["_date"] = date_str
                    turns.append(turn)
            session_idx += 1
    elif isinstance(conversation, list):
        turns = conversation

    items = []
    for idx, turn in enumerate(turns):
        dia_id = turn.get("dia_id", idx)
        text_content = turn.get("text", "").strip()
        speaker_id = turn.get("speaker", turn.get("id", "unknown"))
        session_num = turn.get("_session", turn.get("session", 0))
        date = turn.get("_date", turn.get("date", ""))

        # Map speaker ID to name
        if speaker_id in ("A", "a"):
            speaker = speaker_a
        elif speaker_id in ("B", "b"):
            speaker = speaker_b
        else:
            speaker = speaker_id

        if not text_content:
            continue

        if date:
            full_text = f"[{date}] {speaker}: {text_content}"
        else:
            full_text = f"{speaker}: {text_content}"

        # Store full utterance
        item = make_memory_item(
            full_text,
            source_agent=source_agent,
            source_session=source_session,
            entity=speaker,
            property="utterance",
            value=text_content[:200],
            memory_type="conversation",
            scope="benchmark",
            tags=["locomo", f"session_{session_num}"],
            item_id=f"{source_session}_dia{dia_id}",
        )
        if date:
            item["first_seen"] = date
            item["last_confirmed"] = date
        items.append(item)

        # Add high-precision derived facts linked to the parent utterance.
        # This replaces the old sentence-splitting approach, which created
        # noisy pseudo-facts and diluted retrieval.
        parent_id = item["id"]
        # Use timestamped full_text for extraction so relative dates like
        # "yesterday" / "last year" can be normalized against the dialogue date.
        for fi, fact in enumerate(extract_typed_facts(full_text, speaker=speaker, source_id=parent_id)):
            fact_text = f"{fact.subject} {fact.predicate.replace('_', ' ')}: {fact.value}"
            if date:
                fact_text = f"[{date}] {fact_text}"
            fact_item = make_memory_item(
                fact_text,
                source_agent=source_agent,
                source_session=source_session,
                entity=fact.subject,
                property=fact.predicate,
                value=fact.value[:200],
                memory_type="fact",
                scope="benchmark",
                tags=["locomo", f"session_{session_num}", "typed_fact", fact.fact_type],
                item_id=f"{source_session}_dia{dia_id}_f{fi}",
            )
            fact_item["source_chunk"] = parent_id
            fact_item["confidence"] = fact.confidence
            fact_item["notes"] = json.dumps({
                "fact_type": fact.fact_type,
                "source_id": fact.source_id,
                "evidence_text": fact.evidence_text,
                "qualifiers": fact.qualifiers,
                "extractor": "typed_dialogue_v1",
            }, sort_keys=True)
            if date:
                fact_item["first_seen"] = date
                fact_item["last_confirmed"] = date
            items.append(fact_item)

    return items


def get_qa_pairs(conv_data: dict) -> list[dict]:
    """Extract QA pairs from a LoCoMo conversation entry."""
    import sys as _sys
    _sys.path.insert(0, str(BENCHMARKS_DIR))
    from common import NO_ANSWER_SENTINEL

    qa_list = conv_data.get("qa", conv_data.get("QA", []))
    result = []
    for qa in qa_list:
        question = qa.get("question", qa.get("q", ""))
        category = str(qa.get("category", qa.get("type", "unknown")))
        evidences = qa.get("evidences", qa.get("evidence", []))

        # LoCoMo category 5 = adversarial/UNANSWERABLE. These items have NO
        # `answer` field; the gold lives in `adversarial_answer` and represents the
        # plausible-but-wrong distractor the model must NOT assert. The correct
        # behavior is to abstain, so gold = the no-answer sentinel and we carry the
        # distractor separately for diagnostics. Without this, every cat-5 gold was
        # an empty string and the whole adversarial category was mis-scored.
        if category == "5" or "adversarial_answer" in qa:
            distractor = str(qa.get("adversarial_answer", ""))
            if not question:
                continue
            result.append({
                "question": question,
                "answer": NO_ANSWER_SENTINEL,
                "category": category,
                "unanswerable": True,
                "distractor": distractor,
                "evidences": evidences if isinstance(evidences, list) else [evidences],
            })
            continue

        answer = str(qa.get("answer", qa.get("a", "")))
        if not question or not answer:
            continue

        result.append({
            "question": question,
            "answer": answer,
            "category": category,
            "unanswerable": False,
            "evidences": evidences if isinstance(evidences, list) else [evidences],
        })

    return result
