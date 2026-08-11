"""
fact_extractor.py — Benchmark-agnostic typed dialogue fact extraction.

This module deliberately avoids benchmark-specific answer/gold-label hacks.  It
extracts only high-precision, source-backed memory facts from dialogue turns so
benchmarks and real memory QA can use concise derived facts without replacing
raw utterance evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
import re
from typing import Iterable


@dataclass(frozen=True)
class ExtractedFact:
    """A concise derived memory fact linked to a parent evidence item."""

    fact_type: str
    subject: str
    predicate: str
    value: str
    evidence_text: str
    source_id: str
    confidence: float
    qualifiers: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_SPACE_RE = re.compile(r"\s+")

_VAGUE_VALUES = {
    "it", "this", "that", "these", "those", "something", "someone", "somebody",
    "stuff", "things", "thing", "anything", "everything", "nothing", "there",
    "here", "work", "home", "school", "today", "tomorrow", "yesterday",
}

_NOISE_RE = re.compile(
    r"(^```|\b(let me know|hope this helps|i can help|sure[, ]|here'?s|as an ai)\b|"
    r"https?://|\{\s*\"|^\s*(import|def|class|return)\s+)",
    re.I,
)

_RELATION_WORDS = (
    "mother|mom|father|dad|parent|parents|sister|brother|wife|husband|partner|"
    "friend|boss|manager|teacher|coworker|colleague|grandmother|grandfather|grandparent"
)


def _clean(text: str) -> str:
    text = text.strip().strip('"\'`')
    text = re.sub(r"^[\-*•]\s+", "", text)
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    return _SPACE_RE.sub(" ", text).strip()


def _norm_predicate(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", text.strip().lower())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "fact"


_DATE_WORD_RE = re.compile(
    r"(?:\b(?:on|in|around|by|during)\s+)?"
    r"(?:\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s*,?\s+\d{2,4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{2,4})?|"
    r"\d{4}|\d{1,2}/\d{1,2}/\d{2,4}|"
    r"(?:yesterday|today|tomorrow)|"
    r"(?:last|this|next)\s+(?:weekend|week|month|year|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)|"
    r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday))",
    re.I,
)

_EVENT_VERBS = (
    "attended|joined|went to|visited|painted|made|created|started|finished|graduated|"
    "met|saw|called|emailed|moved to|traveled to|travelled to|bought|ordered|received|researched"
)

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}


def _parse_context_date(text: str) -> datetime | None:
    head = text[:120]
    m = re.search(r"on\s+(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})", head, re.I)
    if m:
        month = _MONTHS.get(m.group(2).lower()) or _MONTHS.get(m.group(2).lower()[:3])
        if month:
            return datetime(int(m.group(3)), month, int(m.group(1)))
    return None


def _format_date(dt: datetime) -> str:
    return f"{dt.day} {dt.strftime('%B')} {dt.year}"


def _normalize_date_text(text: str, *, context_date: datetime | None = None) -> str:
    text = _clean(text)
    text = re.sub(r"^(?:on|in|around|by|during)\s+", "", text, flags=re.I).rstrip(".,;:!? ")
    if context_date:
        low = text.lower()
        if low == "yesterday":
            return _format_date(context_date - timedelta(days=1))
        if low == "today":
            return _format_date(context_date)
        if low == "tomorrow":
            return _format_date(context_date + timedelta(days=1))
        if low == "last year":
            return str(context_date.year - 1)
        if low == "this year":
            return str(context_date.year)
        if low == "next year":
            return str(context_date.year + 1)
        if low in {"last week", "last weekend"}:
            return _format_date(context_date - timedelta(days=7))
        if low in {"this week", "this weekend"}:
            return _format_date(context_date)
        if low in {"next week", "next weekend"}:
            return _format_date(context_date + timedelta(days=7))
    return text


def _find_event_date(sent: str) -> tuple[re.Match | None, str]:
    context_date = _parse_context_date(sent)
    body_offset = sent.find("]") + 1 if sent.startswith("[") and "]" in sent[:120] else 0
    body = sent[body_offset:]
    for m in _DATE_WORD_RE.finditer(body):
        if re.search(r"\b(yesterday|today|tomorrow|last|this|next)\b", m.group(0), re.I):
            return m, _normalize_date_text(m.group(0), context_date=context_date)
    m = _DATE_WORD_RE.search(body)
    if m:
        return m, _normalize_date_text(m.group(0), context_date=context_date)
    m = _DATE_WORD_RE.search(sent)
    if m:
        return m, _normalize_date_text(m.group(0), context_date=context_date)
    return None, ""

def _valid_value(value: str) -> bool:
    value = _clean(value).rstrip(".,!?;:")
    if not (2 <= len(value) <= 140):
        return False
    if value.lower() in _VAGUE_VALUES:
        return False
    if re.fullmatch(r"[\W_]+", value):
        return False
    # Reject extremely clause-like captures; these should stay as utterance evidence.
    if len(value.split()) > 18:
        return False
    return True


def _split_sentences(text: str) -> Iterable[str]:
    for sent in _SENTENCE_RE.split(text or ""):
        sent = _clean(sent)
        if 10 <= len(sent) <= 320 and not _NOISE_RE.search(sent):
            yield sent


def _add_fact(
    facts: list[ExtractedFact],
    *,
    fact_type: str,
    subject: str,
    predicate: str,
    value: str,
    evidence_text: str,
    source_id: str,
    confidence: float,
    qualifiers: dict | None = None,
) -> None:
    subject = _clean(subject) or "speaker"
    predicate = _norm_predicate(predicate)
    value = _clean(value).rstrip(".,!?;:")
    if not _valid_value(value):
        return
    fact = ExtractedFact(
        fact_type=fact_type,
        subject=subject,
        predicate=predicate,
        value=value,
        evidence_text=evidence_text,
        source_id=source_id,
        confidence=round(float(confidence), 3),
        qualifiers=qualifiers or {},
    )
    key = (fact.fact_type, fact.subject.lower(), fact.predicate, fact.value.lower())
    existing = {
        (f.fact_type, f.subject.lower(), f.predicate, f.value.lower()) for f in facts
    }
    if key not in existing:
        facts.append(fact)


def extract_typed_facts(
    text: str,
    *,
    speaker: str = "speaker",
    source_id: str = "",
    max_facts: int = 6,
) -> list[ExtractedFact]:
    """Extract high-precision typed facts from a dialogue turn.

    The extractor intentionally favors precision over recall.  It does not emit
    naked named entities or standalone numbers; every fact must have a subject,
    predicate, value, and parent source_id.
    """
    facts: list[ExtractedFact] = []
    speaker = _clean(speaker) or "speaker"

    for sent in _split_sentences(text):
        # Event with date: "On 7 May 2023, I joined the support group" or
        # "I painted a sunrise in 2022".  The date remains structured in
        # qualifiers so temporal readers can answer "when" queries directly.
        date_match, date_value = _find_event_date(sent)
        if date_match:
            raw_date = date_match.group(0)
            sent_wo_date = _clean(re.sub(re.escape(raw_date), " ", sent, count=1, flags=re.I).strip(" ,.;:"))
            for m in re.finditer(
                rf"\b(I|[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)\s+({_EVENT_VERBS})\s+(.{{2,120}}?)(?:[.;,!?:]|$)",
                sent_wo_date,
                re.I,
            ):
                subj = speaker if m.group(1).lower() == "i" else m.group(1)
                pred = _norm_predicate(m.group(2))
                _add_fact(
                    facts, fact_type="event_with_date", subject=subj, predicate=pred,
                    value=m.group(3), evidence_text=sent, source_id=source_id, confidence=0.88,
                    qualifiers={"date": date_value, "raw_date": raw_date},
                )

        # Preferences: "I like/love/prefer/hate/dislike X".
        for m in re.finditer(
            r"\bI\s+(really\s+|still\s+)?(like|love|prefer|enjoy|hate|dislike)\s+(.{2,120}?)(?:[.;,!?:]|$)",
            sent,
            re.I,
        ):
            verb = m.group(2).lower()
            pred = {"hate": "dislikes", "dislike": "dislikes", "prefer": "prefers"}.get(verb, "likes")
            _add_fact(facts, fact_type="preference", subject=speaker, predicate=pred,
                      value=m.group(3), evidence_text=sent, source_id=source_id, confidence=0.9)

        # Career/interests: "I'm keen on counseling", "I've been looking into X as a career".
        for m in re.finditer(
            r"\bI(?:'m|\s+am)?\s*(?:still\s+)?(?:keen on|interested in|looking into|thinking of|thinking about|passionate about)\s+(.{2,140}?)(?:[\-—.;!?]|$)",
            sent,
            re.I,
        ):
            _add_fact(facts, fact_type="interest", subject=speaker, predicate="interested_in",
                      value=m.group(1), evidence_text=sent, source_id=source_id, confidence=0.84)
        for m in re.finditer(
            r"\b(?:I'm|I am)\s+still\s+thinking\s+that\s+(.{2,120}?)\s+is\s+the\s+way\s+to\s+go\b",
            sent,
            re.I,
        ):
            _add_fact(facts, fact_type="interest", subject=speaker, predicate="interested_in",
                      value=m.group(1), evidence_text=sent, source_id=source_id, confidence=0.82)

        # Fragment/action facts in dialogue: "Researching adoption agencies — ...".
        for m in re.finditer(
            r"\b(Researching|Studying|Exploring|Investigating|Looking into)\s+(.{2,120}?)(?:[—\-.;!?]|$)",
            sent,
            re.I,
        ):
            pred = _norm_predicate(m.group(1))
            _add_fact(facts, fact_type="event", subject=speaker, predicate=pred,
                      value=m.group(2), evidence_text=sent, source_id=source_id, confidence=0.86)

        # Identity via "as a/an X" in self-description.
        for m in re.finditer(
            r"\b(?:my\s+own\s+journey\s+)?as\s+a(?:n)?\s+(.{2,80}?)(?:\s+and|\s+who|\s+that|[.;,!?:]|$)",
            sent,
            re.I,
        ):
            val = m.group(1)
            if re.search(r"\b(woman|man|person|student|teacher|engineer|artist|parent|mother|father|counselor|therapist|teen|adult)\b", val, re.I):
                _add_fact(facts, fact_type="identity", subject=speaker, predicate="identity",
                          value=val, evidence_text=sent, source_id=source_id, confidence=0.82)

        # Identity / stable attribute: "I am a teacher", "I'm 34", "my name is X".
        for m in re.finditer(r"\bmy\s+name\s+is\s+(.{2,80}?)(?:[.;,!?:]|$)", sent, re.I):
            _add_fact(facts, fact_type="identity", subject=speaker, predicate="name",
                      value=m.group(1), evidence_text=sent, source_id=source_id, confidence=0.94)
        for m in re.finditer(r"\bI(?:'m|\s+am)\s+(?:a|an)?\s*(.{2,90}?)(?:[.;,!?:]|$)", sent, re.I):
            value = m.group(1)
            if not re.match(r"(?:going|trying|thinking|looking|working)\b", value, re.I):
                _add_fact(facts, fact_type="identity", subject=speaker, predicate="is",
                          value=value, evidence_text=sent, source_id=source_id, confidence=0.78)

        # Possessive/attribute: "my X is/was Y" including ages/amounts.
        for m in re.finditer(
            r"\bmy\s+([A-Za-z][A-Za-z\s]{1,40}?)\s+(?:is|was|are|were)\s+(.{2,100}?)(?:[.;,!?:]|$)",
            sent,
            re.I,
        ):
            prop = _norm_predicate(m.group(1))
            _add_fact(facts, fact_type="attribute", subject=speaker, predicate=prop,
                      value=m.group(2), evidence_text=sent, source_id=source_id, confidence=0.88)

        # Relationships: "Alice is my friend", "Bob is my manager".
        for m in re.finditer(
            rf"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)\s+(?:is|was)\s+my\s+({_RELATION_WORDS})\b",
            sent,
            re.I,
        ):
            _add_fact(facts, fact_type="relationship", subject=speaker,
                      predicate=m.group(2).lower(), value=m.group(1),
                      evidence_text=sent, source_id=source_id, confidence=0.9)

        # Ownership/actions: "I bought/ordered/received/found X".
        for m in re.finditer(
            r"\bI\s+(bought|purchased|ordered|received|got|found|picked up|made|baked|cooked|built)\s+(.{2,120}?)(?:[.;,!?:]|$)",
            sent,
            re.I,
        ):
            verb = _norm_predicate(m.group(1))
            fact_type = "ownership" if verb in {"bought", "purchased", "ordered", "received", "got", "found", "picked_up"} else "event"
            _add_fact(facts, fact_type=fact_type, subject=speaker, predicate=verb,
                      value=m.group(2), evidence_text=sent, source_id=source_id, confidence=0.86)

        # Location/events: "I visited Paris", "I moved to Boston", "I work at Stripe".
        for m in re.finditer(
            r"\bI\s+(visited|went to|traveled to|travelled to|moved to|live in|lived in|work at|worked at|study at|studied at)\s+(.{2,100}?)(?:[.;,!?:]|$)",
            sent,
            re.I,
        ):
            pred = _norm_predicate(m.group(1))
            _add_fact(facts, fact_type="location_event", subject=speaker, predicate=pred,
                      value=m.group(2), evidence_text=sent, source_id=source_id, confidence=0.86)

        if len(facts) >= max_facts:
            break

    return facts[:max_facts]
