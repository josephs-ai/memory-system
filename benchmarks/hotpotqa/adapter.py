"""
hotpotqa/adapter.py — HotpotQA dataset adapter.

Dataset: https://hotpotqa.github.io/
HuggingFace: hotpot_qa (distractor setting)

HotpotQA tests multi-hop reasoning: answering each question requires
combining information from 2+ documents/paragraphs.

Dataset structure (HuggingFace distractor split):
{
  "id": str,
  "question": str,
  "answer": str,
  "type": str,           # "bridge" or "comparison"
  "level": str,          # "easy", "medium", "hard"
  "supporting_facts": {
    "title": [str],       # paragraph titles that contain supporting evidence
    "sent_id": [int],     # sentence indices within those paragraphs
  },
  "context": {
    "title": [str],       # 10 paragraph titles (2 gold + 8 distractors)
    "sentences": [[str]], # sentences for each paragraph
  },
}

Eval approach:
  - Ingest all 10 context paragraphs as memory items
  - Query with the question
  - Check if answer can be derived from retrieved items
  - Measure whether BOTH supporting paragraphs appear in top-k
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BENCHMARKS_DIR / "datasets" / "hotpotqa"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

LOGGER = logging.getLogger("openclaw.benchmarks.hotpotqa")

LOCAL_PATH = DATASETS_DIR / "hotpotqa_distractor_dev.json"


def download_dataset(force: bool = False) -> Path:
    """Download HotpotQA distractor dev set."""
    if LOCAL_PATH.exists() and not force:
        LOGGER.info("Dataset already exists: %s", LOCAL_PATH)
        return LOCAL_PATH

    LOGGER.info("Downloading HotpotQA dataset...")

    # Try HuggingFace first
    try:
        from datasets import load_dataset as hf_load
        ds = hf_load("hotpot_qa", "distractor", split="validation", trust_remote_code=True)
        data = [dict(item) for item in ds]
        with open(LOCAL_PATH, "w", encoding="utf-8") as f:
            json.dump(data[:500], f)  # Cap at 500 for dev
        LOGGER.info("Saved %d examples to %s", min(500, len(data)), LOCAL_PATH)
        return LOCAL_PATH
    except Exception as e:
        LOGGER.warning("HuggingFace download failed: %s", e)

    # Try direct URL
    try:
        import httpx
        url = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
        with httpx.Client(follow_redirects=True, timeout=120) as client:
            r = client.get(url)
            if r.status_code == 200:
                data = r.json()[:500]
                with open(LOCAL_PATH, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                LOGGER.info("Downloaded %d examples", len(data))
                return LOCAL_PATH
    except Exception as e:
        LOGGER.warning("Direct download failed: %s", e)

    # Synthetic fallback
    LOGGER.warning("All downloads failed. Creating synthetic HotpotQA dataset...")
    data = _create_synthetic_hotpotqa()
    with open(LOCAL_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f)
    LOGGER.info("Created synthetic dataset with %d examples", len(data))
    return LOCAL_PATH


def _create_synthetic_hotpotqa() -> list[dict]:
    """Create synthetic multi-hop QA examples."""
    examples = [
        {
            "id": "synth_bridge_0",
            "question": "What university did the director of Inception attend?",
            "answer": "University College London",
            "type": "bridge",
            "level": "medium",
            "supporting_facts": {"title": ["Inception", "Christopher Nolan"], "sent_id": [0, 1]},
            "context": {
                "title": ["Inception", "Christopher Nolan", "The Dark Knight", "Leonardo DiCaprio",
                          "Warner Bros", "Dream", "Cillian Murphy", "Tom Hardy", "Hans Zimmer", "Interstellar"],
                "sentences": [
                    ["Inception is a 2010 science fiction film directed by Christopher Nolan.",
                     "The film stars Leonardo DiCaprio as a thief who steals secrets through dream invasion."],
                    ["Christopher Nolan is a British-American filmmaker born in London.",
                     "He studied English Literature at University College London.",
                     "He is known for his cerebral and nonlinear storytelling."],
                    ["The Dark Knight is a 2008 superhero film also directed by Christopher Nolan.",
                     "It stars Christian Bale as Batman."],
                    ["Leonardo DiCaprio is an American actor born in Los Angeles.",
                     "He won an Academy Award for The Revenant in 2016."],
                    ["Warner Bros is an American entertainment company.",
                     "It has distributed many of Nolan's films."],
                    ["Dreams are a succession of images and sensations occurring during sleep.",
                     "Lucid dreaming is the awareness of being in a dream."],
                    ["Cillian Murphy is an Irish actor.",
                     "He starred in Peaky Blinders and Oppenheimer."],
                    ["Tom Hardy is an English actor.",
                     "He appeared in Inception and The Dark Knight Rises."],
                    ["Hans Zimmer is a German film composer.",
                     "He composed the score for Inception and Interstellar."],
                    ["Interstellar is a 2014 science fiction film directed by Christopher Nolan.",
                     "It explores themes of time dilation and wormholes."],
                ],
            },
        },
        {
            "id": "synth_bridge_1",
            "question": "In what country was the composer of the Harry Potter film score born?",
            "answer": "United States",
            "type": "bridge",
            "level": "medium",
            "supporting_facts": {"title": ["Harry Potter films", "John Williams"], "sent_id": [1, 0]},
            "context": {
                "title": ["Harry Potter films", "John Williams", "J.K. Rowling", "Daniel Radcliffe",
                          "Hogwarts", "Warner Bros Pictures", "Alan Rickman", "Emma Watson",
                          "Hedwig's Theme", "Film score"],
                "sentences": [
                    ["The Harry Potter film series is based on the novels by J.K. Rowling.",
                     "The musical score for the first three films was composed by John Williams.",
                     "The series was produced by Warner Bros Pictures."],
                    ["John Williams is an American composer born in Floral Park, New York, United States.",
                     "He has composed music for Star Wars, Jaws, and Indiana Jones.",
                     "He has won five Academy Awards for Best Original Score."],
                    ["J.K. Rowling is a British author.",
                     "She wrote the Harry Potter series beginning in 1997."],
                    ["Daniel Radcliffe is an English actor.",
                     "He played Harry Potter in all eight films."],
                    ["Hogwarts is a fictional school of witchcraft and wizardry.",
                     "It is located in Scotland in the Harry Potter universe."],
                    ["Warner Bros Pictures is a film production company.",
                     "It is headquartered in Burbank, California."],
                    ["Alan Rickman was an English actor.",
                     "He portrayed Severus Snape in the Harry Potter films."],
                    ["Emma Watson is a British actress.",
                     "She played Hermione Granger in the Harry Potter series."],
                    ["Hedwig's Theme is the main theme of the Harry Potter films.",
                     "It was composed by John Williams for the first film."],
                    ["A film score is original music composed for a film.",
                     "It is typically performed by an orchestra."],
                ],
            },
        },
        {
            "id": "synth_comparison_0",
            "question": "Are both the Eiffel Tower and the Statue of Liberty made primarily of iron?",
            "answer": "yes",
            "type": "comparison",
            "level": "easy",
            "supporting_facts": {"title": ["Eiffel Tower", "Statue of Liberty"], "sent_id": [1, 1]},
            "context": {
                "title": ["Eiffel Tower", "Statue of Liberty", "Paris", "New York City",
                          "Gustave Eiffel", "Auguste Bartholdi", "Iron", "Steel",
                          "World Heritage Site", "Tourism"],
                "sentences": [
                    ["The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
                     "It was constructed from 7,300 tonnes of puddle iron between 1887 and 1889.",
                     "Gustave Eiffel's engineering company designed and built it."],
                    ["The Statue of Liberty is a colossal neoclassical sculpture on Liberty Island.",
                     "It has a copper exterior but its internal structure is made of iron and steel.",
                     "It was a gift from France to the United States in 1886."],
                    ["Paris is the capital and most populous city of France.",
                     "It is known for landmarks like the Eiffel Tower and the Louvre."],
                    ["New York City is the most populous city in the United States.",
                     "The Statue of Liberty is located in New York Harbor."],
                    ["Gustave Eiffel was a French civil engineer and architect.",
                     "He is best known for the Eiffel Tower."],
                    ["Auguste Bartholdi was a French sculptor.",
                     "He designed the Statue of Liberty."],
                    ["Iron is a chemical element with symbol Fe.",
                     "Wrought iron is an iron alloy with low carbon content."],
                    ["Steel is an alloy of iron and carbon.",
                     "It is stronger than pure iron."],
                    ["A World Heritage Site is a landmark recognized by UNESCO.",
                     "Both the Eiffel Tower and Statue of Liberty are World Heritage Sites."],
                    ["Tourism is travel for recreation or leisure.",
                     "Both landmarks attract millions of visitors annually."],
                ],
            },
        },
        {
            "id": "synth_bridge_2",
            "question": "What is the population of the city where Tesla was founded?",
            "answer": "approximately 1 million",
            "type": "bridge",
            "level": "hard",
            "supporting_facts": {"title": ["Tesla Inc", "San Carlos California"], "sent_id": [0, 1]},
            "context": {
                "title": ["Tesla Inc", "San Carlos California", "Elon Musk", "Electric vehicles",
                          "SpaceX", "Silicon Valley", "Palo Alto", "Battery technology",
                          "Martin Eberhard", "Nikola Tesla"],
                "sentences": [
                    ["Tesla Inc is an American electric vehicle company founded in San Carlos, California in 2003.",
                     "It was incorporated by Martin Eberhard and Marc Tarpenning."],
                    ["San Carlos is a city in San Mateo County, California.",
                     "It has a population of approximately 1 million in the greater metro area.",
                     "It is located in the San Francisco Bay Area."],
                    ["Elon Musk became chairman of Tesla's board in 2004.",
                     "He led the company's Series A funding round."],
                    ["Electric vehicles are powered by electric motors using energy stored in batteries.",
                     "They produce zero direct emissions."],
                    ["SpaceX is an aerospace company also led by Elon Musk.",
                     "It was founded in 2002 in Hawthorne, California."],
                    ["Silicon Valley is a region in Northern California.",
                     "It is a global center for high technology and innovation."],
                    ["Palo Alto is a city adjacent to Stanford University.",
                     "Tesla later moved its headquarters to Palo Alto."],
                    ["Lithium-ion batteries are the primary energy storage for electric vehicles.",
                     "Tesla manufactures batteries at its Gigafactory in Nevada."],
                    ["Martin Eberhard is an American engineer and entrepreneur.",
                     "He co-founded Tesla Motors in 2003."],
                    ["Nikola Tesla was a Serbian-American inventor.",
                     "He is best known for his contributions to alternating current."],
                ],
            },
        },
        {
            "id": "synth_comparison_1",
            "question": "Which was founded first, Apple or Microsoft?",
            "answer": "Microsoft",
            "type": "comparison",
            "level": "easy",
            "supporting_facts": {"title": ["Apple Inc", "Microsoft"], "sent_id": [0, 0]},
            "context": {
                "title": ["Apple Inc", "Microsoft", "Steve Jobs", "Bill Gates",
                          "Personal computers", "Macintosh", "Windows", "IBM",
                          "Steve Wozniak", "Paul Allen"],
                "sentences": [
                    ["Apple Inc was founded on April 1, 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne.",
                     "It is headquartered in Cupertino, California."],
                    ["Microsoft was founded on April 4, 1975 by Bill Gates and Paul Allen.",
                     "It is headquartered in Redmond, Washington."],
                    ["Steve Jobs was an American entrepreneur and inventor.",
                     "He co-founded Apple and served as its CEO until 2011."],
                    ["Bill Gates is an American business magnate.",
                     "He co-founded Microsoft and served as CEO until 2000."],
                    ["Personal computers became popular in the late 1970s and 1980s.",
                     "Both Apple and Microsoft played key roles in this revolution."],
                    ["The Macintosh was introduced by Apple in 1984.",
                     "It was one of the first personal computers with a graphical user interface."],
                    ["Windows is a series of operating systems developed by Microsoft.",
                     "Windows 1.0 was released in 1985."],
                    ["IBM released its first personal computer in 1981.",
                     "It ran Microsoft's MS-DOS operating system."],
                    ["Steve Wozniak is an American engineer.",
                     "He designed the Apple I and Apple II computers."],
                    ["Paul Allen was an American business magnate.",
                     "He co-founded Microsoft with Bill Gates."],
                ],
            },
        },
        {
            "id": "synth_bridge_3",
            "question": "What genre is the band that recorded Abbey Road?",
            "answer": "rock",
            "type": "bridge",
            "level": "easy",
            "supporting_facts": {"title": ["Abbey Road album", "The Beatles"], "sent_id": [0, 0]},
            "context": {
                "title": ["Abbey Road album", "The Beatles", "Abbey Road Studios", "George Martin",
                          "Let It Be", "John Lennon", "Paul McCartney", "George Harrison",
                          "Ringo Starr", "British Invasion"],
                "sentences": [
                    ["Abbey Road is the eleventh studio album by the English rock band the Beatles.",
                     "It was released on 26 September 1969 by Apple Records."],
                    ["The Beatles were an English rock band formed in Liverpool in 1960.",
                     "They are widely regarded as the most influential band of all time.",
                     "Their primary genre was rock music."],
                    ["Abbey Road Studios is a recording studio in London.",
                     "It is famous for its association with the Beatles."],
                    ["George Martin was an English record producer.",
                     "He produced most of the Beatles' recordings."],
                    ["Let It Be was the final album released by the Beatles.",
                     "It came out in May 1970."],
                    ["John Lennon was an English musician and singer-songwriter.",
                     "He was a co-lead vocalist and rhythm guitarist of the Beatles."],
                    ["Paul McCartney is an English musician.",
                     "He was the bass guitarist and co-lead vocalist of the Beatles."],
                    ["George Harrison was an English musician.",
                     "He was the lead guitarist of the Beatles."],
                    ["Ringo Starr is an English musician.",
                     "He was the drummer of the Beatles."],
                    ["The British Invasion was a cultural movement in the mid-1960s.",
                     "British rock bands became popular in the United States."],
                ],
            },
        },
        {
            "id": "synth_bridge_4",
            "question": "What is the capital of the country where the Nobel Prize ceremony is held?",
            "answer": "Stockholm",
            "type": "bridge",
            "level": "medium",
            "supporting_facts": {"title": ["Nobel Prize", "Sweden"], "sent_id": [1, 0]},
            "context": {
                "title": ["Nobel Prize", "Sweden", "Alfred Nobel", "Stockholm Concert Hall",
                          "Norway", "Royal Swedish Academy", "Dynamite", "Marie Curie",
                          "Nobel Peace Prize", "Chemistry"],
                "sentences": [
                    ["The Nobel Prize is a set of annual international awards.",
                     "The ceremonies for physics, chemistry, medicine, literature, and economics are held in Sweden.",
                     "They were established by the will of Alfred Nobel in 1895."],
                    ["Sweden is a Nordic country in Northern Europe.",
                     "Its capital and largest city is Stockholm.",
                     "It has a population of about 10 million."],
                    ["Alfred Nobel was a Swedish chemist and engineer.",
                     "He invented dynamite and held 355 patents."],
                    ["The Stockholm Concert Hall hosts the Nobel Prize ceremony.",
                     "It is located in the Hötorget area of central Stockholm."],
                    ["Norway hosts the Nobel Peace Prize ceremony.",
                     "It is held in Oslo City Hall."],
                    ["The Royal Swedish Academy of Sciences selects Nobel laureates.",
                     "It was founded in 1739."],
                    ["Dynamite is an explosive invented by Alfred Nobel in 1867.",
                     "Its invention made Nobel very wealthy."],
                    ["Marie Curie won Nobel Prizes in both physics and chemistry.",
                     "She is the only person to win in two different sciences."],
                    ["The Nobel Peace Prize is awarded in Oslo, Norway.",
                     "It recognizes efforts for peace and diplomacy."],
                    ["Chemistry is the study of matter and its properties.",
                     "The Nobel Prize in Chemistry has been awarded since 1901."],
                ],
            },
        },
        {
            "id": "synth_bridge_5",
            "question": "Who painted the ceiling of the chapel where papal conclaves are held?",
            "answer": "Michelangelo",
            "type": "bridge",
            "level": "medium",
            "supporting_facts": {"title": ["Papal conclave", "Sistine Chapel"], "sent_id": [0, 1]},
            "context": {
                "title": ["Papal conclave", "Sistine Chapel", "Michelangelo", "Vatican City",
                          "Pope Francis", "St Peter's Basilica", "Renaissance art", "Raphael",
                          "The Last Judgment", "Rome"],
                "sentences": [
                    ["A papal conclave is a meeting of the College of Cardinals to elect a new pope.",
                     "Since 1492, conclaves have been held in the Sistine Chapel."],
                    ["The Sistine Chapel is a chapel in the Apostolic Palace in Vatican City.",
                     "Its ceiling was painted by Michelangelo between 1508 and 1512.",
                     "The ceiling depicts scenes from the Book of Genesis."],
                    ["Michelangelo di Lodovico Buonarroti Simoni was an Italian sculptor and painter.",
                     "He is considered one of the greatest artists of all time."],
                    ["Vatican City is an independent city-state within Rome.",
                     "It is the smallest internationally recognized independent state."],
                    ["Pope Francis is the current head of the Catholic Church.",
                     "He was elected in a papal conclave in March 2013."],
                    ["St Peter's Basilica is a church in Vatican City.",
                     "It is the largest church in the world by interior area."],
                    ["Renaissance art emerged in Italy in the 14th century.",
                     "It emphasized humanism and realistic representation."],
                    ["Raphael was an Italian painter contemporary with Michelangelo.",
                     "He painted the School of Athens in the Vatican."],
                    ["The Last Judgment is a fresco by Michelangelo.",
                     "It covers the altar wall of the Sistine Chapel."],
                    ["Rome is the capital city of Italy.",
                     "Vatican City is an enclave within Rome."],
                ],
            },
        },
        {
            "id": "synth_comparison_2",
            "question": "Is Mount Everest taller than K2?",
            "answer": "yes",
            "type": "comparison",
            "level": "easy",
            "supporting_facts": {"title": ["Mount Everest", "K2"], "sent_id": [0, 0]},
            "context": {
                "title": ["Mount Everest", "K2", "Himalayas", "Karakoram",
                          "Edmund Hillary", "Mountaineering", "Nepal", "Pakistan",
                          "Tenzing Norgay", "Eight-thousander"],
                "sentences": [
                    ["Mount Everest is Earth's highest mountain above sea level at 8,849 meters.",
                     "It is located in the Mahalangur Himal sub-range of the Himalayas."],
                    ["K2 is the second highest mountain in the world at 8,611 meters.",
                     "It is located on the China-Pakistan border in the Karakoram range."],
                    ["The Himalayas is a mountain range in Asia.",
                     "It contains many of the world's highest peaks."],
                    ["The Karakoram is a mountain range spanning Pakistan, China, and India.",
                     "It contains four of the world's fourteen eight-thousanders."],
                    ["Edmund Hillary was a New Zealand mountaineer.",
                     "He and Tenzing Norgay were the first to summit Everest in 1953."],
                    ["Mountaineering is the sport of climbing mountains.",
                     "High-altitude mountaineering involves climbing above 8,000 meters."],
                    ["Nepal is a landlocked country in South Asia.",
                     "Mount Everest sits on the border of Nepal and Tibet."],
                    ["Pakistan is a country in South Asia.",
                     "K2 is located in the Gilgit-Baltistan region of Pakistan."],
                    ["Tenzing Norgay was a Nepali-Indian Sherpa mountaineer.",
                     "He reached the summit of Everest with Edmund Hillary."],
                    ["An eight-thousander is a mountain over 8,000 meters.",
                     "There are fourteen such peaks in the world."],
                ],
            },
        },
        {
            "id": "synth_bridge_6",
            "question": "What language is primarily spoken in the country where the Amazon River originates?",
            "answer": "Spanish",
            "type": "bridge",
            "level": "medium",
            "supporting_facts": {"title": ["Amazon River", "Peru"], "sent_id": [1, 1]},
            "context": {
                "title": ["Amazon River", "Peru", "Brazil", "Amazon Rainforest",
                          "Andes", "South America", "Nile", "Congo River",
                          "Lima", "Iquitos"],
                "sentences": [
                    ["The Amazon River is the largest river by discharge volume in the world.",
                     "It originates in the Andes mountains of Peru.",
                     "The river flows through Brazil before emptying into the Atlantic Ocean."],
                    ["Peru is a country in western South America.",
                     "The official language of Peru is Spanish.",
                     "Its capital city is Lima."],
                    ["Brazil is the largest country in South America.",
                     "The official language of Brazil is Portuguese."],
                    ["The Amazon Rainforest is the world's largest tropical rainforest.",
                     "It covers much of the Amazon basin in South America."],
                    ["The Andes is the longest continental mountain range in the world.",
                     "It runs along the western coast of South America."],
                    ["South America is a continent in the Western Hemisphere.",
                     "It contains twelve sovereign states."],
                    ["The Nile is a major river in northeastern Africa.",
                     "It is traditionally considered the longest river in the world."],
                    ["The Congo River is the second-largest river in Africa.",
                     "It flows through the Democratic Republic of Congo."],
                    ["Lima is the capital and largest city of Peru.",
                     "It was founded by Spanish conquistador Francisco Pizarro in 1535."],
                    ["Iquitos is the largest city in the Peruvian Amazon.",
                     "It is only accessible by river or air."],
                ],
            },
        },
    ]
    return examples


def load_dataset(max_examples: int | None = None) -> list[dict]:
    """Load HotpotQA dataset."""
    if not LOCAL_PATH.exists():
        download_dataset()

    with open(LOCAL_PATH, encoding="utf-8") as f:
        data = json.load(f)

    if max_examples is not None:
        data = data[:max_examples]

    LOGGER.info("Loaded %d HotpotQA examples", len(data))
    return data


def context_to_memory_items(
    example: dict,
    *,
    source_agent: str,
    source_session: str,
) -> list[dict]:
    """Convert context paragraphs to memory items."""
    import sys
    sys.path.insert(0, str(BENCHMARKS_DIR))
    from common import make_memory_item

    items = []
    titles = example.get("context", {}).get("title", [])
    sentences_list = example.get("context", {}).get("sentences", [])

    for idx, (title, sents) in enumerate(zip(titles, sentences_list)):
        text = f"{title}: {' '.join(sents)}"
        item = make_memory_item(
            text,
            source_agent=source_agent,
            source_session=source_session,
            entity=title,
            property="paragraph",
            value=text[:200],
            memory_type="fact",
            scope="benchmark",
            tags=["hotpotqa", f"para_{idx}", title],
            item_id=f"{source_session}_para{idx}",
        )
        items.append(item)

    return items


def get_supporting_titles(example: dict) -> set[str]:
    """Get the set of titles that contain supporting facts."""
    sf = example.get("supporting_facts", {})
    return set(sf.get("title", []))
