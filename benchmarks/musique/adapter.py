"""
musique/adapter.py — MuSiQue dataset adapter.

Dataset: https://github.com/StonyBrookNLP/musique
HuggingFace: StonyBrookNLP/musique

MuSiQue tests multi-step compositional reasoning (2-4 hops).
Each question decomposes into sub-questions that must be answered sequentially.

Dataset structure:
{
  "id": str,
  "question": str,
  "answer": str,
  "answer_aliases": [str],
  "decomposition": [
    {"id": int, "question": str, "answer": str, "paragraph_support_idx": int}
  ],
  "paragraphs": [
    {"idx": int, "title": str, "paragraph_text": str, "is_supporting": bool}
  ],
  "question_decomposition": [...]  # same as decomposition
}

Eval approach:
  - Ingest all paragraphs as memory items
  - Query with the top-level question
  - Check if supporting paragraphs are retrieved
  - Measure hop coverage (did we find evidence for each reasoning step?)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BENCHMARKS_DIR / "datasets" / "musique"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

LOGGER = logging.getLogger("openclaw.benchmarks.musique")

LOCAL_PATH = DATASETS_DIR / "musique_dev.json"


def download_dataset(force: bool = False) -> Path:
    """Download MuSiQue dataset."""
    if LOCAL_PATH.exists() and not force:
        return LOCAL_PATH

    LOGGER.info("Downloading MuSiQue dataset...")

    try:
        from datasets import load_dataset as hf_load
        ds = hf_load("StonyBrookNLP/musique", "answerable", split="validation",
                      trust_remote_code=True)
        data = [dict(item) for item in ds]
        with open(LOCAL_PATH, "w", encoding="utf-8") as f:
            json.dump(data[:500], f)
        LOGGER.info("Saved %d examples", min(500, len(data)))
        return LOCAL_PATH
    except Exception as e:
        LOGGER.warning("HuggingFace download failed: %s", e)

    # Synthetic fallback
    LOGGER.warning("Creating synthetic MuSiQue dataset...")
    data = _create_synthetic_musique()
    with open(LOCAL_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f)
    LOGGER.info("Created %d synthetic examples", len(data))
    return LOCAL_PATH


def _create_synthetic_musique() -> list[dict]:
    """Create synthetic multi-hop compositional reasoning examples (2-4 hops)."""
    examples = [
        # 2-hop examples
        {
            "id": "synth_2hop_0",
            "question": "What is the capital of the country where the inventor of the telephone was born?",
            "answer": "Edinburgh",
            "answer_aliases": ["Edinburgh, Scotland"],
            "decomposition": [
                {"id": 0, "question": "Who invented the telephone?",
                 "answer": "Alexander Graham Bell", "paragraph_support_idx": 0},
                {"id": 1, "question": "Where was Alexander Graham Bell born?",
                 "answer": "Edinburgh, Scotland", "paragraph_support_idx": 1},
            ],
            "paragraphs": [
                {"idx": 0, "title": "Telephone", "is_supporting": True,
                 "paragraph_text": "The telephone was invented by Alexander Graham Bell in 1876. He received the first patent for the device on March 7, 1876."},
                {"idx": 1, "title": "Alexander Graham Bell", "is_supporting": True,
                 "paragraph_text": "Alexander Graham Bell was born on March 3, 1847 in Edinburgh, Scotland. He later moved to Canada and then the United States."},
                {"idx": 2, "title": "Thomas Edison", "is_supporting": False,
                 "paragraph_text": "Thomas Edison was an American inventor born in Milan, Ohio. He invented the phonograph and the practical incandescent light bulb."},
                {"idx": 3, "title": "Scotland", "is_supporting": False,
                 "paragraph_text": "Scotland is a country that is part of the United Kingdom. Edinburgh is the capital city of Scotland."},
                {"idx": 4, "title": "Elisha Gray", "is_supporting": False,
                 "paragraph_text": "Elisha Gray was an American electrical engineer who co-founded Western Electric. He filed a patent caveat for a telephone design on the same day as Bell."},
            ],
        },
        {
            "id": "synth_2hop_1",
            "question": "What river flows through the city where Mozart was born?",
            "answer": "Salzach",
            "answer_aliases": ["Salzach River"],
            "decomposition": [
                {"id": 0, "question": "Where was Mozart born?",
                 "answer": "Salzburg", "paragraph_support_idx": 0},
                {"id": 1, "question": "What river flows through Salzburg?",
                 "answer": "Salzach", "paragraph_support_idx": 1},
            ],
            "paragraphs": [
                {"idx": 0, "title": "Wolfgang Amadeus Mozart", "is_supporting": True,
                 "paragraph_text": "Wolfgang Amadeus Mozart was born on January 27, 1756 in Salzburg, Austria. He was a prolific and influential composer of the Classical period."},
                {"idx": 1, "title": "Salzburg", "is_supporting": True,
                 "paragraph_text": "Salzburg is a city in Austria on the border of Germany. The Salzach River flows through the city center, dividing it into old and new sections."},
                {"idx": 2, "title": "Ludwig van Beethoven", "is_supporting": False,
                 "paragraph_text": "Beethoven was born in Bonn, Germany in 1770. He moved to Vienna in 1792 and remained there for the rest of his life."},
                {"idx": 3, "title": "Vienna", "is_supporting": False,
                 "paragraph_text": "Vienna is the capital of Austria. The Danube River flows through the city. It was a major center for classical music."},
                {"idx": 4, "title": "Danube River", "is_supporting": False,
                 "paragraph_text": "The Danube is the second-longest river in Europe. It flows through ten countries including Germany, Austria, and Hungary."},
            ],
        },
        # 3-hop examples
        {
            "id": "synth_3hop_0",
            "question": "What language is spoken in the country that borders the nation where the Mona Lisa is displayed?",
            "answer": "German",
            "answer_aliases": ["German language"],
            "decomposition": [
                {"id": 0, "question": "Where is the Mona Lisa displayed?",
                 "answer": "Louvre Museum in Paris, France", "paragraph_support_idx": 0},
                {"id": 1, "question": "What country borders France to the east?",
                 "answer": "Germany", "paragraph_support_idx": 1},
                {"id": 2, "question": "What language is spoken in Germany?",
                 "answer": "German", "paragraph_support_idx": 2},
            ],
            "paragraphs": [
                {"idx": 0, "title": "Mona Lisa", "is_supporting": True,
                 "paragraph_text": "The Mona Lisa is a portrait painted by Leonardo da Vinci. It is displayed at the Louvre Museum in Paris, France. It is the most visited artwork in the world."},
                {"idx": 1, "title": "France geography", "is_supporting": True,
                 "paragraph_text": "France is a country in Western Europe. It borders Belgium and Luxembourg to the north, Germany and Switzerland to the east, Italy to the southeast, and Spain to the southwest."},
                {"idx": 2, "title": "Germany", "is_supporting": True,
                 "paragraph_text": "Germany is a country in Central Europe. The official language is German. Its capital is Berlin and it has a population of about 83 million."},
                {"idx": 3, "title": "Louvre Museum", "is_supporting": False,
                 "paragraph_text": "The Louvre is the world's largest art museum located in Paris. It was originally a royal palace before becoming a museum in 1793."},
                {"idx": 4, "title": "Leonardo da Vinci", "is_supporting": False,
                 "paragraph_text": "Leonardo da Vinci was an Italian polymath of the Renaissance. He was born in Vinci, Italy in 1452."},
                {"idx": 5, "title": "Spanish language", "is_supporting": False,
                 "paragraph_text": "Spanish is a Romance language spoken in Spain and most of Latin America. It has over 500 million native speakers worldwide."},
            ],
        },
        {
            "id": "synth_3hop_1",
            "question": "What ocean borders the country where the company that makes the iPhone is headquartered?",
            "answer": "Pacific Ocean",
            "answer_aliases": ["Pacific"],
            "decomposition": [
                {"id": 0, "question": "What company makes the iPhone?",
                 "answer": "Apple", "paragraph_support_idx": 0},
                {"id": 1, "question": "Where is Apple headquartered?",
                 "answer": "Cupertino, California, United States", "paragraph_support_idx": 1},
                {"id": 2, "question": "What ocean borders the western United States?",
                 "answer": "Pacific Ocean", "paragraph_support_idx": 2},
            ],
            "paragraphs": [
                {"idx": 0, "title": "iPhone", "is_supporting": True,
                 "paragraph_text": "The iPhone is a smartphone made by Apple Inc. It was first released on June 29, 2007. It revolutionized the mobile phone industry."},
                {"idx": 1, "title": "Apple Inc", "is_supporting": True,
                 "paragraph_text": "Apple Inc is an American technology company headquartered in Cupertino, California, United States. It was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976."},
                {"idx": 2, "title": "United States geography", "is_supporting": True,
                 "paragraph_text": "The United States is bordered by the Atlantic Ocean to the east and the Pacific Ocean to the west. It also borders Canada to the north and Mexico to the south."},
                {"idx": 3, "title": "Samsung", "is_supporting": False,
                 "paragraph_text": "Samsung is a South Korean conglomerate headquartered in Seoul. It is one of the world's largest producers of smartphones."},
                {"idx": 4, "title": "California", "is_supporting": False,
                 "paragraph_text": "California is a state on the west coast of the United States. It is the most populous state with nearly 40 million residents."},
                {"idx": 5, "title": "Atlantic Ocean", "is_supporting": False,
                 "paragraph_text": "The Atlantic Ocean is the second largest ocean in the world. It separates the Americas from Europe and Africa."},
            ],
        },
        # 4-hop example
        {
            "id": "synth_4hop_0",
            "question": "What is the currency of the country where the author of Harry Potter was born, whose capital hosted the 2012 Olympics?",
            "answer": "pound sterling",
            "answer_aliases": ["British pound", "GBP", "pound"],
            "decomposition": [
                {"id": 0, "question": "Who wrote Harry Potter?",
                 "answer": "J.K. Rowling", "paragraph_support_idx": 0},
                {"id": 1, "question": "Where was J.K. Rowling born?",
                 "answer": "Yate, England, United Kingdom", "paragraph_support_idx": 1},
                {"id": 2, "question": "What city hosted the 2012 Olympics in the United Kingdom?",
                 "answer": "London", "paragraph_support_idx": 2},
                {"id": 3, "question": "What is the currency of the United Kingdom?",
                 "answer": "pound sterling", "paragraph_support_idx": 3},
            ],
            "paragraphs": [
                {"idx": 0, "title": "Harry Potter series", "is_supporting": True,
                 "paragraph_text": "Harry Potter is a series of fantasy novels written by British author J.K. Rowling. The first novel was published in 1997 by Bloomsbury."},
                {"idx": 1, "title": "J.K. Rowling", "is_supporting": True,
                 "paragraph_text": "Joanne Rowling, known by her pen name J.K. Rowling, was born on 31 July 1965 in Yate, Gloucestershire, England, United Kingdom."},
                {"idx": 2, "title": "2012 Summer Olympics", "is_supporting": True,
                 "paragraph_text": "The 2012 Summer Olympics were held in London, the capital of the United Kingdom, from 27 July to 12 August 2012."},
                {"idx": 3, "title": "United Kingdom economy", "is_supporting": True,
                 "paragraph_text": "The United Kingdom uses the pound sterling as its currency. The Bank of England is the central bank responsible for issuing banknotes."},
                {"idx": 4, "title": "Hogwarts", "is_supporting": False,
                 "paragraph_text": "Hogwarts School of Witchcraft and Wizardry is a fictional school in the Harry Potter series. It is located in Scotland."},
                {"idx": 5, "title": "2008 Olympics", "is_supporting": False,
                 "paragraph_text": "The 2008 Summer Olympics were held in Beijing, China. It was the first time China hosted the Summer Olympics."},
                {"idx": 6, "title": "Euro currency", "is_supporting": False,
                 "paragraph_text": "The euro is the official currency of 20 of the 27 member states of the European Union. It was introduced in 1999."},
            ],
        },
        # More 2-hop examples for volume
        {
            "id": "synth_2hop_2",
            "question": "What sport is played at the stadium where the team managed by Pep Guardiola plays?",
            "answer": "football",
            "answer_aliases": ["soccer", "association football"],
            "decomposition": [
                {"id": 0, "question": "What team does Pep Guardiola manage?",
                 "answer": "Manchester City", "paragraph_support_idx": 0},
                {"id": 1, "question": "What stadium does Manchester City play at?",
                 "answer": "Etihad Stadium", "paragraph_support_idx": 1},
            ],
            "paragraphs": [
                {"idx": 0, "title": "Pep Guardiola", "is_supporting": True,
                 "paragraph_text": "Pep Guardiola is a Spanish football manager who currently manages Manchester City in the English Premier League. He previously managed Barcelona and Bayern Munich."},
                {"idx": 1, "title": "Manchester City", "is_supporting": True,
                 "paragraph_text": "Manchester City Football Club plays its home football matches at the Etihad Stadium in Manchester, England. The stadium has a capacity of 53,400."},
                {"idx": 2, "title": "Manchester United", "is_supporting": False,
                 "paragraph_text": "Manchester United plays at Old Trafford stadium. It is one of the most successful clubs in English football."},
                {"idx": 3, "title": "Etihad Stadium", "is_supporting": False,
                 "paragraph_text": "The Etihad Stadium was originally built for the 2002 Commonwealth Games. It was later converted for football use."},
                {"idx": 4, "title": "Premier League", "is_supporting": False,
                 "paragraph_text": "The Premier League is the top level of the English football league system. It was founded in 1992."},
            ],
        },
        {
            "id": "synth_2hop_3",
            "question": "What is the population of the country where sushi originated?",
            "answer": "125 million",
            "answer_aliases": ["about 125 million", "approximately 125 million"],
            "decomposition": [
                {"id": 0, "question": "Where did sushi originate?",
                 "answer": "Japan", "paragraph_support_idx": 0},
                {"id": 1, "question": "What is the population of Japan?",
                 "answer": "approximately 125 million", "paragraph_support_idx": 1},
            ],
            "paragraphs": [
                {"idx": 0, "title": "Sushi", "is_supporting": True,
                 "paragraph_text": "Sushi is a traditional Japanese dish consisting of prepared vinegared rice with seafood, vegetables, and sometimes tropical fruits. It originated in Japan and has become popular worldwide."},
                {"idx": 1, "title": "Japan demographics", "is_supporting": True,
                 "paragraph_text": "Japan is an island country in East Asia with a population of approximately 125 million people. It is the eleventh most populous country in the world."},
                {"idx": 2, "title": "Chinese cuisine", "is_supporting": False,
                 "paragraph_text": "Chinese cuisine encompasses a diverse range of cooking styles from different regions of China. Dim sum and Peking duck are popular dishes."},
                {"idx": 3, "title": "Rice cultivation", "is_supporting": False,
                 "paragraph_text": "Rice is the staple food for more than half of the world's population. It is grown primarily in Asia."},
                {"idx": 4, "title": "South Korea", "is_supporting": False,
                 "paragraph_text": "South Korea has a population of about 52 million. Its capital is Seoul and it is known for K-pop and technology."},
            ],
        },
        {
            "id": "synth_3hop_2",
            "question": "What is the timezone of the city where the headquarters of the company founded by Jeff Bezos is located?",
            "answer": "Pacific Time",
            "answer_aliases": ["PST", "Pacific Standard Time", "PT"],
            "decomposition": [
                {"id": 0, "question": "What company did Jeff Bezos found?",
                 "answer": "Amazon", "paragraph_support_idx": 0},
                {"id": 1, "question": "Where is Amazon headquartered?",
                 "answer": "Seattle, Washington", "paragraph_support_idx": 1},
                {"id": 2, "question": "What timezone is Seattle in?",
                 "answer": "Pacific Time", "paragraph_support_idx": 2},
            ],
            "paragraphs": [
                {"idx": 0, "title": "Jeff Bezos", "is_supporting": True,
                 "paragraph_text": "Jeff Bezos is an American entrepreneur who founded Amazon in 1994 in his garage in Bellevue, Washington. He served as CEO until 2021."},
                {"idx": 1, "title": "Amazon headquarters", "is_supporting": True,
                 "paragraph_text": "Amazon is headquartered in Seattle, Washington. Its campus includes the Amazon Spheres, three glass conservatories in the Denny Triangle neighborhood."},
                {"idx": 2, "title": "Seattle", "is_supporting": True,
                 "paragraph_text": "Seattle is a seaport city in the Pacific Northwest. It operates in the Pacific Time zone and is the largest city in Washington state."},
                {"idx": 3, "title": "Elon Musk", "is_supporting": False,
                 "paragraph_text": "Elon Musk is CEO of Tesla and SpaceX. He was born in Pretoria, South Africa."},
                {"idx": 4, "title": "New York", "is_supporting": False,
                 "paragraph_text": "New York City is in the Eastern Time zone. It is the most populous city in the United States."},
                {"idx": 5, "title": "Bellevue", "is_supporting": False,
                 "paragraph_text": "Bellevue is a city in Washington state across Lake Washington from Seattle. It is part of the Seattle metropolitan area."},
            ],
        },
    ]
    return examples


def load_dataset(max_examples: int | None = None) -> list[dict]:
    """Load MuSiQue dataset."""
    if not LOCAL_PATH.exists():
        download_dataset()

    with open(LOCAL_PATH, encoding="utf-8") as f:
        data = json.load(f)

    if max_examples is not None:
        data = data[:max_examples]

    LOGGER.info("Loaded %d MuSiQue examples", len(data))
    return data


def paragraphs_to_memory_items(
    example: dict,
    *,
    source_agent: str,
    source_session: str,
) -> list[dict]:
    """Convert paragraphs to memory items."""
    import sys
    sys.path.insert(0, str(BENCHMARKS_DIR))
    from common import make_memory_item

    items = []
    for para in example.get("paragraphs", []):
        title = para.get("title", "unknown")
        text = f"{title}: {para['paragraph_text']}"
        item = make_memory_item(
            text,
            source_agent=source_agent,
            source_session=source_session,
            entity=title,
            property="paragraph",
            value=para["paragraph_text"][:200],
            memory_type="fact",
            scope="benchmark",
            tags=["musique", f"para_{para['idx']}"],
            item_id=f"{source_session}_para{para['idx']}",
        )
        items.append(item)
    return items


def get_supporting_indices(example: dict) -> set[int]:
    """Get paragraph indices that are supporting."""
    return {p["idx"] for p in example.get("paragraphs", []) if p.get("is_supporting")}


def get_hop_count(example: dict) -> int:
    """Get the number of reasoning hops."""
    return len(example.get("decomposition", []))
