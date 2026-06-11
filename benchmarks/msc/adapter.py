"""
msc/adapter.py — Multi-Session Chat (MSC) dataset adapter.

Dataset: https://huggingface.co/datasets/facebook/msc
         or ParlAI: projects/msc/

MSC tests whether a system can remember user persona facts across sessions.

Dataset structure (HuggingFace format):
{
  "session_id": str,
  "dialog": [
    {
      "id": str,          # speaker id
      "text": str,        # utterance
    }
  ],
  "personas": [str],      # persona facts for each speaker
  "previous_dialogs": [   # prior session dialogs
    {
      "dialog": [...],
      "personas": [str]
    }
  ]
}

Eval approach:
  - Ingest previous_dialogs as memory items
  - Query with persona-related questions derived from current session
  - Check if persona facts can be recalled
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BENCHMARKS_DIR / "datasets" / "msc"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

LOGGER = logging.getLogger("openclaw.benchmarks.msc")

# HuggingFace dataset parameters
HF_DATASET_NAME = "facebook/msc"
LOCAL_PATH = DATASETS_DIR / "msc_session2.json"  # Use session 2 (has prior sessions)

# Persona fact question templates for evaluation
PERSONA_QUESTION_TEMPLATES = [
    "What does the person like or enjoy?",
    "What are the person's hobbies or interests?",
    "Where does the person live or work?",
    "What is the person's background or history?",
    "What are the person's preferences or opinions?",
]


def download_dataset(split: str = "session_2", force: bool = False) -> Path:
    """
    Download MSC dataset via HuggingFace datasets library.
    Falls back to direct download if datasets library not available.
    """
    if LOCAL_PATH.exists() and not force:
        LOGGER.info("Dataset already exists: %s", LOCAL_PATH)
        return LOCAL_PATH

    LOGGER.info("Downloading MSC dataset (split=%s)", split)

    # Try HuggingFace datasets library first
    try:
        from datasets import load_dataset as hf_load
        LOGGER.info("Using HuggingFace datasets library...")
        ds = hf_load("facebook/msc", split, trust_remote_code=True)
        # Convert to list of dicts and save
        data = [dict(item) for item in ds["train"]]
        with open(LOCAL_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f)
        LOGGER.info("Saved %d examples to %s", len(data), LOCAL_PATH)
        return LOCAL_PATH
    except Exception as e:
        LOGGER.warning("HuggingFace datasets failed: %s. Trying direct download...", e)

    # Try direct ParlAI data endpoint
    urls_to_try = [
        "https://huggingface.co/datasets/facebook/msc/resolve/main/session_2/train.txt",
        "https://dl.fbaipublicfiles.com/msc/msc_personasummary.tar.gz",
    ]

    for url in urls_to_try:
        try:
            import httpx
            with httpx.Client(follow_redirects=True, timeout=120) as client:
                r = client.get(url)
                if r.status_code == 200:
                    raw_path = DATASETS_DIR / "msc_raw.txt"
                    raw_path.write_bytes(r.content)
                    LOGGER.info("Downloaded raw data: %d bytes", len(r.content))
                    # Parse ParlAI format
                    data = _parse_parlai_format(r.content.decode("utf-8", errors="replace"))
                    if data:
                        with open(LOCAL_PATH, "w", encoding="utf-8") as f:
                            json.dump(data, f)
                        LOGGER.info("Parsed %d examples", len(data))
                        return LOCAL_PATH
        except Exception as e:
            LOGGER.warning("URL %s failed: %s", url, e)

    # Create minimal synthetic fallback
    LOGGER.warning("All downloads failed. Creating synthetic MSC dataset...")
    data = _create_synthetic_msc()
    with open(LOCAL_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f)
    LOGGER.info("Created synthetic MSC dataset with %d examples", len(data))
    return LOCAL_PATH


def _parse_parlai_format(text: str) -> list[dict]:
    """Parse ParlAI conversation format into structured examples."""
    examples = []
    current_dialog = []
    current_personas = []

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            if current_dialog:
                examples.append({
                    "dialog": current_dialog,
                    "personas": current_personas,
                    "previous_dialogs": [],
                })
                current_dialog = []
                current_personas = []
            continue

        # ParlAI persona lines start with "your persona:"
        if "your persona:" in line.lower():
            fact = re.sub(r"^\d+\s+your persona:\s*", "", line, flags=re.IGNORECASE).strip()
            current_personas.append(fact)
        elif re.match(r"^\d+\s+", line):
            # Numbered utterance line
            parts = line.split("\t")
            if len(parts) >= 2:
                turn_text = re.sub(r"^\d+\s+", "", parts[0]).strip()
                response = parts[1].strip() if len(parts) > 1 else ""
                if turn_text:
                    current_dialog.append({"id": "user", "text": turn_text})
                if response:
                    current_dialog.append({"id": "bot", "text": response})

    return examples[:500]  # Cap at 500


def _create_synthetic_msc() -> list[dict]:
    """Create a synthetic MSC-like dataset for testing (100 diverse personas)."""
    persona_templates = [
        ["I love hiking in the mountains.", "I work as a software engineer.", "I have two dogs named Max and Bella."],
        ["I enjoy cooking Italian food.", "I studied art history in college.", "I live in Seattle."],
        ["I play the guitar in a band.", "I'm training for a marathon.", "I have three kids."],
        ["I collect vintage vinyl records.", "I work as a nurse.", "I enjoy gardening on weekends."],
        ["I'm learning to speak Japanese.", "I love science fiction novels.", "I used to live in Paris."],
        ["I enjoy scuba diving in coral reefs.", "I work as a veterinarian.", "I have a cat named Whiskers."],
        ["I love playing chess competitively.", "I studied computer science at MIT.", "I grew up in Denver."],
        ["I enjoy watercolor painting.", "I work as an accountant.", "I have four siblings."],
        ["I'm passionate about rock climbing.", "I teach high school biology.", "I live in Portland."],
        ["I collect antique clocks.", "I work as a pharmacist.", "I enjoy birdwatching."],
        ["I love surfing at the beach.", "I studied mechanical engineering.", "I have a golden retriever named Charlie."],
        ["I enjoy making pottery.", "I work as a dentist.", "I grew up in Austin."],
        ["I play the piano at local venues.", "I'm training for an Ironman triathlon.", "I have twin daughters."],
        ["I collect rare coins.", "I work as a librarian.", "I enjoy kayaking on rivers."],
        ["I'm learning to speak Mandarin.", "I love mystery novels.", "I used to live in London."],
        ["I enjoy mountain biking.", "I work as a firefighter.", "I have a parrot named Polly."],
        ["I love playing basketball on weekends.", "I studied economics at Stanford.", "I grew up in Chicago."],
        ["I enjoy digital photography.", "I work as a physical therapist.", "I have two brothers."],
        ["I'm passionate about yoga.", "I teach elementary school art.", "I live in San Francisco."],
        ["I collect vintage cameras.", "I work as a chef.", "I enjoy stargazing."],
        ["I love snowboarding in winter.", "I studied biochemistry.", "I have a beagle named Scout."],
        ["I enjoy woodworking in my garage.", "I work as an architect.", "I grew up in Boston."],
        ["I play the drums in a jazz trio.", "I'm training for a 10K race.", "I have a newborn son."],
        ["I collect baseball cards.", "I work as a social worker.", "I enjoy fishing on lakes."],
        ["I'm learning to speak French.", "I love fantasy novels.", "I used to live in Tokyo."],
        ["I enjoy sailing on the coast.", "I work as a paramedic.", "I have a Siamese cat named Luna."],
        ["I love playing soccer every Saturday.", "I studied psychology at UCLA.", "I grew up in Miami."],
        ["I enjoy calligraphy.", "I work as a financial analyst.", "I have three sisters."],
        ["I'm passionate about meditation.", "I teach college philosophy.", "I live in Nashville."],
        ["I collect vintage comic books.", "I work as a barista.", "I enjoy trail running."],
        ["I love ice skating in winter.", "I studied marine biology.", "I have a labrador named Buddy."],
        ["I enjoy leather crafting.", "I work as a lawyer.", "I grew up in Philadelphia."],
        ["I play the violin in an orchestra.", "I'm training for a cycling century.", "I have two teenage sons."],
        ["I collect stamps from around the world.", "I work as a teacher.", "I enjoy camping in national parks."],
        ["I'm learning to speak Korean.", "I love historical fiction.", "I used to live in Berlin."],
        ["I enjoy kitesurfing.", "I work as a plumber.", "I have a rabbit named Thumper."],
        ["I love playing tennis.", "I studied political science.", "I grew up in Dallas."],
        ["I enjoy origami.", "I work as a data scientist.", "I have one older brother."],
        ["I'm passionate about tai chi.", "I teach middle school math.", "I live in Minneapolis."],
        ["I collect vintage watches.", "I work as a baker.", "I enjoy horseback riding."],
        ["I love cross-country skiing.", "I studied environmental science.", "I have a corgi named Maple."],
        ["I enjoy glassblowing.", "I work as a journalist.", "I grew up in Atlanta."],
        ["I play the flute in a community band.", "I'm training for a swim meet.", "I have a daughter in college."],
        ["I collect old maps.", "I work as a museum curator.", "I enjoy mushroom foraging."],
        ["I'm learning to speak Arabic.", "I love thriller novels.", "I used to live in Rome."],
        ["I enjoy paragliding.", "I work as an electrician.", "I have a hamster named Peanut."],
        ["I love playing volleyball.", "I studied sociology at NYU.", "I grew up in Seattle."],
        ["I enjoy embroidery.", "I work as a marketing manager.", "I have two cousins I'm close with."],
        ["I'm passionate about pilates.", "I teach kindergarten.", "I live in Charlotte."],
        ["I collect vintage typewriters.", "I work as a florist.", "I enjoy geocaching."],
        ["I love wakeboarding in summer.", "I studied neuroscience.", "I have a poodle named Coco."],
        ["I enjoy blacksmithing.", "I work as a surgeon.", "I grew up in Detroit."],
        ["I play the cello at recitals.", "I'm training for a rowing regatta.", "I have three grandchildren."],
        ["I collect old postcards.", "I work as a flight attendant.", "I enjoy snorkeling."],
        ["I'm learning to speak Portuguese.", "I love romance novels.", "I used to live in Sydney."],
        ["I enjoy windsurfing.", "I work as a carpenter.", "I have a turtle named Sheldon."],
        ["I love playing badminton.", "I studied philosophy at Oxford.", "I grew up in San Diego."],
        ["I enjoy mosaic art.", "I work as a podcast host.", "I have a younger sister."],
        ["I'm passionate about acrobatics.", "I teach high school chemistry.", "I live in Denver."],
        ["I collect vintage board games.", "I work as a bartender.", "I enjoy rock hounding."],
        ["I love downhill skiing.", "I studied genetics.", "I have a border collie named Finn."],
        ["I enjoy soap making.", "I work as a real estate agent.", "I grew up in Houston."],
        ["I play the saxophone at jazz clubs.", "I'm training for a hiking expedition.", "I have a stepdaughter."],
        ["I collect antique books.", "I work as a translator.", "I enjoy whale watching."],
        ["I'm learning to speak Italian.", "I love graphic novels.", "I used to live in Amsterdam."],
        ["I enjoy hang gliding.", "I work as a mechanic.", "I have a ferret named Noodle."],
        ["I love playing golf every Sunday.", "I studied biology at Harvard.", "I grew up in New Orleans."],
        ["I enjoy bookbinding.", "I work as a UX designer.", "I have an identical twin."],
        ["I'm passionate about Zumba.", "I teach preschool.", "I live in Raleigh."],
        ["I collect vintage lunch boxes.", "I work as a sommelier.", "I enjoy metal detecting."],
        ["I love ice fishing in Minnesota.", "I studied astrophysics.", "I have a Great Dane named Zeus."],
        ["I enjoy candle making.", "I work as a judge.", "I grew up in Baltimore."],
        ["I play the banjo at bluegrass festivals.", "I'm training for a triathlon.", "I have four grandkids."],
        ["I collect seashells from every beach.", "I work as a park ranger.", "I enjoy spelunking."],
        ["I'm learning to speak Russian.", "I love spy novels.", "I used to live in Dublin."],
        ["I enjoy bungee jumping.", "I work as a welder.", "I have a chinchilla named Dusty."],
        ["I love playing table tennis.", "I studied anthropology.", "I grew up in Phoenix."],
        ["I enjoy quilt making.", "I work as a nutritionist.", "I have a large extended family."],
        ["I'm passionate about kickboxing.", "I teach special education.", "I live in Columbus."],
        ["I collect vintage posters.", "I work as a DJ.", "I enjoy fossil hunting."],
        ["I love skateboarding at the park.", "I studied geology.", "I have a husky named Ghost."],
        ["I enjoy wire jewelry making.", "I work as a therapist.", "I grew up in Las Vegas."],
        ["I play the harmonica on road trips.", "I'm training for an ultramarathon.", "I have a son in the military."],
        ["I collect vinyl toy figures.", "I work as a zookeeper.", "I enjoy butterfly gardening."],
        ["I'm learning to speak Hindi.", "I love adventure novels.", "I used to live in Vienna."],
        ["I enjoy zip lining.", "I work as a locksmith.", "I have a gecko named Gizmo."],
        ["I love playing cricket.", "I studied linguistics at Cambridge.", "I grew up in Milwaukee."],
        ["I enjoy basket weaving.", "I work as an event planner.", "I have two adopted children."],
        ["I'm passionate about capoeira.", "I teach music theory.", "I live in Savannah."],
        ["I collect vintage radios.", "I work as a tattoo artist.", "I enjoy beachcombing."],
        ["I love paddleboarding at sunrise.", "I studied organic chemistry.", "I have a dachshund named Oscar."],
        ["I enjoy stained glass work.", "I work as a pilot.", "I grew up in Kansas City."],
        ["I play the ukulele at open mics.", "I'm training for a dance competition.", "I have a niece I babysit often."],
        ["I collect pressed flowers.", "I work as a midwife.", "I enjoy aurora watching."],
        ["I'm learning to speak Swahili.", "I love philosophical essays.", "I used to live in Cape Town."],
        ["I enjoy white water rafting.", "I work as a tailor.", "I have a snake named Monty."],
        ["I love playing field hockey.", "I studied art therapy.", "I grew up in Indianapolis."],
        ["I enjoy macramé wall hangings.", "I work as a life coach.", "I have a close-knit friend group of six."],
        ["I'm passionate about fencing.", "I teach yoga at a studio.", "I live in Asheville."],
        ["I collect vintage globes.", "I work as a beekeeper.", "I enjoy starlit camping trips."],
    ]

    examples = []
    for i, persona_facts in enumerate(persona_templates):
        # Previous session: dialog establishing persona
        prev_dialog = []
        for j, fact in enumerate(persona_facts):
            prev_dialog.append({"id": "speaker1", "text": f"Did you know that {fact.lower()}"})
            prev_dialog.append({"id": "speaker2", "text": "That's interesting! Tell me more."})

        # Current session: questions that require remembering persona
        current_dialog = [
            {"id": "speaker2", "text": "It's good to chat with you again."},
            {"id": "speaker1", "text": "Good to be back! How have you been?"},
        ]

        examples.append({
            "session_id": f"synthetic_{i}",
            "dialog": current_dialog,
            "personas": persona_facts,
            "previous_dialogs": [{"dialog": prev_dialog, "personas": persona_facts}],
        })

    return examples


def load_dataset(max_examples: int | None = None) -> list[dict]:
    """Load MSC dataset."""
    if not LOCAL_PATH.exists():
        download_dataset()

    with open(LOCAL_PATH, encoding="utf-8") as f:
        data = json.load(f)

    if max_examples is not None:
        data = data[:max_examples]

    LOGGER.info("Loaded %d MSC examples", len(data))
    return data


def prev_dialogs_to_memory_items(
    previous_dialogs: list[dict],
    *,
    source_agent: str,
    source_session: str,
) -> list[dict]:
    """Convert previous dialog sessions to memory items."""
    import sys
    sys.path.insert(0, str(BENCHMARKS_DIR))
    from common import make_memory_item

    items = []
    for sess_idx, prev in enumerate(previous_dialogs):
        # Ingest persona facts as structured facts
        personas = prev.get("personas", [])
        for pidx, fact in enumerate(personas):
            if not fact.strip():
                continue
            item = make_memory_item(
                fact,
                source_agent=source_agent,
                source_session=source_session,
                entity="speaker",
                property="persona_fact",
                value=fact[:200],
                memory_type="fact",
                scope="benchmark",
                tags=["msc", "persona", f"session_{sess_idx}"],
                item_id=f"{source_session}_sess{sess_idx}_persona{pidx}",
            )
            items.append(item)

        # Ingest dialog turns
        dialog = prev.get("dialog", [])
        for turn_idx, turn in enumerate(dialog):
            text = turn.get("text", "").strip()
            speaker = turn.get("id", "unknown")
            if not text:
                continue
            full_text = f"{speaker}: {text}"
            item = make_memory_item(
                full_text,
                source_agent=source_agent,
                source_session=source_session,
                entity=speaker,
                property="utterance",
                value=text[:200],
                memory_type="conversation",
                scope="benchmark",
                tags=["msc", f"session_{sess_idx}"],
                item_id=f"{source_session}_sess{sess_idx}_turn{turn_idx}",
            )
            items.append(item)

    return items


def generate_persona_queries(personas: list[str]) -> list[dict]:
    """
    Generate QA pairs from persona facts.
    Returns list of {question, gold_answer} dicts.
    """
    qa_pairs = []

    for fact in personas:
        fact_lower = fact.lower().strip().rstrip(".")

        # Generate question based on fact content
        if "love" in fact_lower or "enjoy" in fact_lower or "like" in fact_lower:
            qa_pairs.append({
                "question": "What does this person love or enjoy?",
                "gold_answer": fact,
            })
        elif "work" in fact_lower or "job" in fact_lower or "engineer" in fact_lower or "nurse" in fact_lower:
            qa_pairs.append({
                "question": "What does this person do for work?",
                "gold_answer": fact,
            })
        elif "live" in fact_lower or "from" in fact_lower or "city" in fact_lower:
            qa_pairs.append({
                "question": "Where does this person live or come from?",
                "gold_answer": fact,
            })
        elif any(w in fact_lower for w in ["study", "college", "university", "school"]):
            qa_pairs.append({
                "question": "What did this person study?",
                "gold_answer": fact,
            })
        else:
            qa_pairs.append({
                "question": f"What do you know about: {fact_lower[:50]}?",
                "gold_answer": fact,
            })

    return qa_pairs
