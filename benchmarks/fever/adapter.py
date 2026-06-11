"""
fever/adapter.py — FEVER (Fact Extraction and VERification) benchmark adapter.

Dataset: https://fever.ai/
HuggingFace: fever/fever

FEVER tests whether a system can verify claims against stored evidence.
Each claim is labeled SUPPORTS, REFUTES, or NOT ENOUGH INFO.

Dataset structure:
{
  "id": int,
  "label": str,           # "SUPPORTS", "REFUTES", "NOT ENOUGH INFO"
  "claim": str,
  "evidence": [            # list of evidence groups (any one is sufficient)
    [
      [annotation_id, evidence_id, wiki_title, sentence_idx]
    ]
  ],
}

Eval approach:
  - Ingest a knowledge base of facts (from evidence wiki pages)
  - Present claims and ask the system to verify
  - Measure if the system retrieves relevant evidence
  - Score verification accuracy (SUPPORTS/REFUTES/NEI)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BENCHMARKS_DIR / "datasets" / "fever"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

LOGGER = logging.getLogger("openclaw.benchmarks.fever")

LOCAL_PATH = DATASETS_DIR / "fever_dev.json"


def download_dataset(force: bool = False) -> Path:
    """Download FEVER dataset."""
    if LOCAL_PATH.exists() and not force:
        return LOCAL_PATH

    LOGGER.info("Downloading FEVER dataset...")

    try:
        from datasets import load_dataset as hf_load
        ds = hf_load("fever", "v1.0", split="labelled_dev", trust_remote_code=True)
        data = [dict(item) for item in ds]
        with open(LOCAL_PATH, "w", encoding="utf-8") as f:
            json.dump(data[:500], f)
        LOGGER.info("Saved %d examples", min(500, len(data)))
        return LOCAL_PATH
    except Exception as e:
        LOGGER.warning("HuggingFace download failed: %s", e)

    LOGGER.warning("Creating synthetic FEVER dataset...")
    data = _create_synthetic_fever()
    with open(LOCAL_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f)
    LOGGER.info("Created %d synthetic examples", len(data))
    return LOCAL_PATH


def _create_synthetic_fever() -> list[dict]:
    """Create synthetic fact verification examples."""
    examples = [
        # --- SUPPORTS ---
        {
            "id": 1,
            "claim": "The Eiffel Tower is located in Paris.",
            "label": "SUPPORTS",
            "evidence_texts": [
                {"title": "Eiffel Tower", "text": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France."},
            ],
            "distractor_texts": [
                {"title": "Big Ben", "text": "Big Ben is the nickname for the Great Bell of the striking clock at the Palace of Westminster in London."},
                {"title": "Colosseum", "text": "The Colosseum is an oval amphitheatre in the centre of the city of Rome, Italy."},
            ],
        },
        {
            "id": 2,
            "claim": "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
            "label": "SUPPORTS",
            "evidence_texts": [
                {"title": "Boiling point", "text": "The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure of 1 atm."},
            ],
            "distractor_texts": [
                {"title": "Freezing point", "text": "Water freezes at 0 degrees Celsius (32 degrees Fahrenheit) at standard pressure."},
                {"title": "Triple point", "text": "The triple point of water is at 0.01 degrees Celsius and 611.73 pascals."},
            ],
        },
        {
            "id": 3,
            "claim": "Albert Einstein developed the theory of general relativity.",
            "label": "SUPPORTS",
            "evidence_texts": [
                {"title": "General relativity", "text": "General relativity is a theory of gravitation developed by Albert Einstein between 1907 and 1915."},
            ],
            "distractor_texts": [
                {"title": "Isaac Newton", "text": "Isaac Newton formulated the laws of motion and universal gravitation in the 17th century."},
                {"title": "Quantum mechanics", "text": "Quantum mechanics was developed by multiple physicists including Max Planck and Niels Bohr."},
            ],
        },
        {
            "id": 4,
            "claim": "The Great Wall of China is visible from space with the naked eye.",
            "label": "REFUTES",
            "evidence_texts": [
                {"title": "Great Wall of China", "text": "The Great Wall of China is a series of fortifications stretching over 13,000 miles. Contrary to popular belief, it is not visible from space with the naked eye, as confirmed by multiple astronauts."},
            ],
            "distractor_texts": [
                {"title": "Space observation", "text": "Many large structures on Earth are visible from low Earth orbit, including cities and highways."},
                {"title": "Chinese history", "text": "The Great Wall was built over many centuries, with the most well-known sections built during the Ming Dynasty."},
            ],
        },
        {
            "id": 5,
            "claim": "Humans only use 10 percent of their brains.",
            "label": "REFUTES",
            "evidence_texts": [
                {"title": "Brain myths", "text": "The claim that humans only use 10 percent of their brains is a myth. Neuroimaging studies show that virtually all areas of the brain are active and have known functions."},
            ],
            "distractor_texts": [
                {"title": "Neuroscience", "text": "The human brain contains approximately 86 billion neurons connected by trillions of synapses."},
                {"title": "Brain anatomy", "text": "The brain is divided into regions including the cerebrum, cerebellum, and brainstem."},
            ],
        },
        {
            "id": 6,
            "claim": "Lightning never strikes the same place twice.",
            "label": "REFUTES",
            "evidence_texts": [
                {"title": "Lightning", "text": "Lightning frequently strikes the same place repeatedly. Tall structures like the Empire State Building are struck about 20-25 times per year."},
            ],
            "distractor_texts": [
                {"title": "Thunder", "text": "Thunder is the sound caused by lightning. The distance can be estimated by counting seconds between the flash and the sound."},
            ],
        },
        {
            "id": 7,
            "claim": "The Amazon River is the longest river in the world.",
            "label": "REFUTES",
            "evidence_texts": [
                {"title": "Nile River", "text": "The Nile River is traditionally considered the longest river in the world at approximately 6,650 km, though some measurements suggest the Amazon may be longer."},
                {"title": "Amazon River", "text": "The Amazon River is the largest river by discharge volume. Its length is approximately 6,400 km, making it the second longest after the Nile."},
            ],
            "distractor_texts": [
                {"title": "Mississippi River", "text": "The Mississippi River is the second longest river in North America at 2,320 miles."},
            ],
        },
        {
            "id": 8,
            "claim": "Shakespeare wrote Romeo and Juliet.",
            "label": "SUPPORTS",
            "evidence_texts": [
                {"title": "Romeo and Juliet", "text": "Romeo and Juliet is a tragedy written by William Shakespeare early in his career, believed to have been written between 1591 and 1596."},
            ],
            "distractor_texts": [
                {"title": "Hamlet", "text": "Hamlet is a tragedy written by William Shakespeare. It is his longest play."},
                {"title": "Christopher Marlowe", "text": "Christopher Marlowe was an English playwright contemporary with Shakespeare."},
            ],
        },
        {
            "id": 9,
            "claim": "The speed of light is approximately 300,000 kilometers per second.",
            "label": "SUPPORTS",
            "evidence_texts": [
                {"title": "Speed of light", "text": "The speed of light in vacuum is exactly 299,792,458 metres per second, approximately 300,000 km/s."},
            ],
            "distractor_texts": [
                {"title": "Speed of sound", "text": "The speed of sound in air is approximately 343 meters per second at 20 degrees Celsius."},
            ],
        },
        {
            "id": 10,
            "claim": "Mount Kilimanjaro is in Kenya.",
            "label": "REFUTES",
            "evidence_texts": [
                {"title": "Mount Kilimanjaro", "text": "Mount Kilimanjaro is a dormant volcano in Tanzania. It is the highest mountain in Africa at 5,895 meters above sea level."},
            ],
            "distractor_texts": [
                {"title": "Kenya", "text": "Kenya is a country in East Africa known for its wildlife reserves and the Great Rift Valley."},
                {"title": "Tanzania", "text": "Tanzania is an East African country known for Serengeti National Park and Mount Kilimanjaro."},
            ],
        },
        # --- NOT ENOUGH INFO ---
        {
            "id": 11,
            "claim": "The CEO of Acme Corp graduated from Harvard University.",
            "label": "NOT ENOUGH INFO",
            "evidence_texts": [
                {"title": "Acme Corp", "text": "Acme Corp is a technology company founded in 2015. It develops cloud infrastructure software."},
            ],
            "distractor_texts": [
                {"title": "Harvard University", "text": "Harvard University is a private Ivy League research university in Cambridge, Massachusetts."},
            ],
        },
        {
            "id": 12,
            "claim": "The population of Mars will exceed 1 million by 2100.",
            "label": "NOT ENOUGH INFO",
            "evidence_texts": [
                {"title": "Mars colonization", "text": "Various organizations including SpaceX have proposed plans for Mars colonization. No permanent human settlement currently exists on Mars."},
            ],
            "distractor_texts": [
                {"title": "Mars facts", "text": "Mars is the fourth planet from the Sun with a thin atmosphere composed primarily of carbon dioxide."},
            ],
        },
        # More SUPPORTS
        {
            "id": 13,
            "claim": "DNA has a double helix structure.",
            "label": "SUPPORTS",
            "evidence_texts": [
                {"title": "DNA structure", "text": "DNA has a double helix structure, first described by James Watson and Francis Crick in 1953, based on X-ray diffraction data from Rosalind Franklin."},
            ],
            "distractor_texts": [
                {"title": "RNA", "text": "RNA is typically single-stranded unlike DNA. It plays various roles in coding and gene expression."},
            ],
        },
        {
            "id": 14,
            "claim": "The Earth orbits the Sun.",
            "label": "SUPPORTS",
            "evidence_texts": [
                {"title": "Earth orbit", "text": "Earth orbits the Sun at an average distance of about 150 million kilometers, taking approximately 365.25 days to complete one orbit."},
            ],
            "distractor_texts": [
                {"title": "Moon orbit", "text": "The Moon orbits the Earth at an average distance of about 384,400 kilometers."},
            ],
        },
        {
            "id": 15,
            "claim": "Goldfish have a three-second memory.",
            "label": "REFUTES",
            "evidence_texts": [
                {"title": "Goldfish memory", "text": "The idea that goldfish have a three-second memory is a myth. Studies have shown that goldfish can remember things for months and can be trained to respond to stimuli."},
            ],
            "distractor_texts": [
                {"title": "Fish cognition", "text": "Fish are more intelligent than commonly believed. Some species can use tools and recognize individual faces."},
            ],
        },
    ]
    return examples


def load_dataset(max_examples: int | None = None) -> list[dict]:
    """Load FEVER dataset."""
    if not LOCAL_PATH.exists():
        download_dataset()

    with open(LOCAL_PATH, encoding="utf-8") as f:
        data = json.load(f)

    if max_examples is not None:
        data = data[:max_examples]

    LOGGER.info("Loaded %d FEVER examples", len(data))
    return data


def evidence_to_memory_items(
    example: dict,
    *,
    source_agent: str,
    source_session: str,
) -> list[dict]:
    """Convert evidence + distractors to memory items."""
    import sys
    sys.path.insert(0, str(BENCHMARKS_DIR))
    from common import make_memory_item

    items = []
    idx = 0

    # Evidence paragraphs
    for ev in example.get("evidence_texts", []):
        text = f"{ev['title']}: {ev['text']}"
        item = make_memory_item(
            text,
            source_agent=source_agent,
            source_session=source_session,
            entity=ev["title"],
            property="knowledge",
            value=ev["text"][:200],
            memory_type="fact",
            scope="benchmark",
            tags=["fever", "evidence"],
            item_id=f"{source_session}_ev{idx}",
        )
        items.append(item)
        idx += 1

    # Distractor paragraphs
    for dist in example.get("distractor_texts", []):
        text = f"{dist['title']}: {dist['text']}"
        item = make_memory_item(
            text,
            source_agent=source_agent,
            source_session=source_session,
            entity=dist["title"],
            property="knowledge",
            value=dist["text"][:200],
            memory_type="fact",
            scope="benchmark",
            tags=["fever", "distractor"],
            item_id=f"{source_session}_dist{idx}",
        )
        items.append(item)
        idx += 1

    return items
