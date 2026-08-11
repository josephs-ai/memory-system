"""
reader.py — Intent-routed memory QA reader with structured answer contract.

Key ideas:
- Separates retrieval from reading: retrieval stays LLM-free; reader is optional.
- Routes to different reader strategies based on query intent.
- Emits a structured answer contract: answer, answer_type, confidence, evidence_ids, abstain.
- Includes a concise answer postprocessor to improve F1-style metrics without
  benchmark-specific hacking.
- Falls back to heuristic extraction when no LLM reader is configured.

No benchmark-specific logic lives here.  This module is intentionally general-purpose.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Literal

LOGGER = logging.getLogger("openclaw.benchmarks.reader")

# ---------------------------------------------------------------------------
# Answer contract
# ---------------------------------------------------------------------------

AnswerType = Literal[
    "entity",
    "noun_phrase",
    "date",
    "count",
    "list",
    "boolean",
    "explanation",
    "unknown",
]


@dataclass
class AnswerResult:
    """Structured output of the reader."""

    answer: str
    answer_type: AnswerType = "noun_phrase"
    confidence: float = 0.0
    evidence_ids: list[str] = field(default_factory=list)
    abstain: bool = False
    reader_mode: str = "heuristic"

    def to_dict(self) -> dict:
        return asdict(self)


_ABSTAIN = AnswerResult(
    answer="unknown",
    answer_type="unknown",
    confidence=0.0,
    abstain=True,
    reader_mode="abstained",
)


# ---------------------------------------------------------------------------
# Intent → reader strategy routing
# ---------------------------------------------------------------------------

_INTENT_TO_STRATEGY = {
    "aggregation_count": "aggregation",
    "temporal_when": "fact_lookup",
    "temporal_latest": "fact_lookup",
    "temporal_earliest": "fact_lookup",
    "temporal_range": "fact_lookup",
    "preference": "fact_lookup",
    "direct_fact": "fact_lookup",
    "contradiction_or_update": "fact_lookup",
    "multi_hop": "reasoning",
    "open_ended_summary": "summary",
    "unknown": "fact_lookup",
}

# ---------------------------------------------------------------------------
# Context builder — build a clean evidence pack from retrieved items
# ---------------------------------------------------------------------------

_VALUE_RE = re.compile(r"\s+")


def _item_to_context_line(item: dict, rank: int) -> str:
    entity = item.get("entity") or ""
    prop = (item.get("property") or "").replace("_", " ")
    val = (item.get("value") or "").strip()
    text = (item.get("text") or "").strip()
    ts = item.get("last_confirmed") or item.get("first_seen") or ""
    # ts may arrive as a datetime from the DB layer; coerce to an ISO-ish string
    # so the [:10] date slice (YYYY-MM-DD) works regardless of source type.
    if ts and not isinstance(ts, str):
        ts = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)

    parts = []
    if entity and prop:
        parts.append(f"[{entity} · {prop}]")
    elif entity:
        parts.append(f"[{entity}]")
    if ts:
        parts.append(f"({ts[:10]})")
    if val:
        parts.append(val[:320])
    elif text:
        parts.append(text[:320])
    return f"{rank}. {' '.join(parts)}" if parts else ""


def _build_context(retrieved_items: list[dict], max_items: int = 20) -> str:
    lines = []
    for i, item in enumerate(retrieved_items[:max_items], start=1):
        line = _item_to_context_line(item, i)
        if line:
            lines.append(line)
    return "\n".join(lines)


def _evidence_ids(retrieved_items: list[dict], max_items: int = 20) -> list[str]:
    return [
        str(item.get("id") or item.get("item_id") or "")
        for item in retrieved_items[:max_items]
        if item.get("id") or item.get("item_id")
    ]


@dataclass(frozen=True)
class AggregationRow:
    """Normalized evidence row for count/list aggregation."""

    key: str
    label: str
    evidence_id: str
    entity: str = ""
    property: str = ""
    session: str = ""
    timestamp: str = ""

    def to_context_line(self, rank: int) -> str:
        pieces = [f"{rank}.", self.label]
        meta = []
        if self.entity:
            meta.append(f"entity={self.entity}")
        if self.property:
            meta.append(f"property={self.property}")
        if self.session:
            meta.append(f"session={self.session}")
        if self.timestamp:
            meta.append(f"date={self.timestamp[:10]}")
        if self.evidence_id:
            meta.append(f"id={self.evidence_id}")
        if meta:
            pieces.append(f"[{'; '.join(meta)}]")
        return " ".join(pieces)


def _row_label(item: dict) -> str:
    val = (item.get("value") or "").strip()
    text = (item.get("text") or "").strip()
    if val:
        return _VALUE_RE.sub(" ", val)[:180]
    return _VALUE_RE.sub(" ", text)[:180]


def _row_session(item: dict) -> str:
    tags = item.get("tags") or []
    if isinstance(tags, list):
        for tag in tags:
            if isinstance(tag, str) and tag.startswith("session_"):
                return tag
    return str(item.get("source_session") or "")


def build_aggregation_table(retrieved_items: list[dict], max_rows: int = 80) -> list[AggregationRow]:
    """Build a deduped, normalized table for count/list questions.

    This is not benchmark-specific.  It groups by normalized value/text plus
    entity/property/session so repeated vector hits for the same fact do not
    inflate counts, while preserving evidence IDs for citation.
    """
    rows: list[AggregationRow] = []
    seen: set[str] = set()
    for item in retrieved_items[:max_rows]:
        label = _row_label(item)
        if not label:
            continue
        entity = str(item.get("entity") or "")
        prop = str(item.get("property") or "")
        session = _row_session(item)
        timestamp = str(item.get("last_confirmed") or item.get("first_seen") or "")
        evidence_id = str(item.get("id") or item.get("item_id") or "")
        norm_label = re.sub(r"[^a-z0-9]+", " ", label.lower()).strip()
        key = "|".join([entity.lower(), prop.lower(), session.lower(), norm_label])
        if key in seen:
            continue
        seen.add(key)
        rows.append(AggregationRow(
            key=key,
            label=label,
            evidence_id=evidence_id,
            entity=entity,
            property=prop,
            session=session,
            timestamp=timestamp,
        ))
    return rows


def _build_aggregation_context(retrieved_items: list[dict], max_rows: int = 80) -> tuple[str, list[AggregationRow]]:
    rows = build_aggregation_table(retrieved_items, max_rows=max_rows)
    return "\n".join(row.to_context_line(i) for i, row in enumerate(rows, start=1)), rows


# ---------------------------------------------------------------------------
# Heuristic reader (zero LLM cost fallback)
# ---------------------------------------------------------------------------

_NOISE_LEAD_RE = re.compile(
    r"^(alice|bob|user|speaker|assistant)?\s*(said|mentioned|replied|told you|answered)\s*(that|:)?\s*",
    re.I,
)
_QUOTE_RE = re.compile(r'^["\'](.+)["\']$')


def _strip_verbose_lead(text: str) -> str:
    """Strip conversational lead-in from heuristic answer candidates."""
    m = _NOISE_LEAD_RE.match(text.strip())
    if m:
        text = text[m.end():].strip()
    m = _QUOTE_RE.match(text.strip())
    if m:
        text = m.group(1).strip()
    return text.lstrip(".,;:").strip()


def _parse_notes(item: dict) -> dict:
    notes = item.get("notes") or {}
    if isinstance(notes, dict):
        return notes
    if isinstance(notes, str) and notes.strip():
        try:
            return json.loads(notes)
        except Exception:
            return {}
    return {}


def _date_from_item(item: dict) -> str | None:
    notes = _parse_notes(item)
    qualifiers = notes.get("qualifiers") or item.get("qualifiers") or {}
    if isinstance(qualifiers, str):
        try:
            qualifiers = json.loads(qualifiers)
        except Exception:
            qualifiers = {}
    for key in ("date", "raw_date", "when"):
        val = qualifiers.get(key) if isinstance(qualifiers, dict) else None
        if val:
            return str(val).strip()
    for key in ("event_date", "date"):
        val = item.get(key)
        if val:
            return str(val).strip()
    return None


_FIRST_PERSON_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bI\s+graduated\s+with\s+(?:a\s+)?degree\s+in\s+([^,.!?;]+)", re.I), "noun_phrase"),
    (re.compile(r"\bI\s+(?:have|had|earned|received)\s+(?:a\s+)?degree\s+in\s+([^,.!?;]+)", re.I), "noun_phrase"),
    (re.compile(r"\bI\s+(?:work|worked)\s+as\s+(?:a|an)?\s*([^,.!?;]+)", re.I), "noun_phrase"),
    (re.compile(r"\bI\s+(?:work|worked)\s+at\s+([^,.!?;]+)", re.I), "entity"),
    (re.compile(r"\bI\s+(?:live|lived)\s+in\s+([^,.!?;]+)", re.I), "entity"),
    (re.compile(r"\bI\s+(?:bought|ordered|received|got|found|made|built|painted)\s+([^,.!?;]+)", re.I), "noun_phrase"),
    (re.compile(r"\bI\s+(?:prefer|like|love|enjoy|hate|dislike)\s+([^,.!?;]+)", re.I), "noun_phrase"),
]


def _extract_first_person_answer(text: str) -> tuple[str, AnswerType] | None:
    text = _strip_verbose_lead(text)
    for pattern, answer_type in _FIRST_PERSON_PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        ans = m.group(1).strip().rstrip(".,;:!?")
        ans = re.sub(r"\s+(?:which|that|because|and)\b.*$", "", ans, flags=re.I).strip()
        if 2 <= len(ans) <= 100:
            return ans, answer_type  # type: ignore[return-value]
    return None


def _heuristic_answer(question: str, retrieved_items: list[dict], intent: str) -> AnswerResult:
    """Produce a concise heuristic answer using retrieved items."""
    if not retrieved_items:
        return _ABSTAIN

    if intent == "temporal_when":
        for item in retrieved_items[:10]:
            date = _date_from_item(item)
            if date:
                return AnswerResult(
                    answer=date,
                    answer_type="date",
                    confidence=float(item.get("confidence") or item.get("score") or 0.75),
                    evidence_ids=_evidence_ids([item]),
                    reader_mode="heuristic_temporal_date",
                )

    if intent == "aggregation_count":
        rows = build_aggregation_table(retrieved_items)
        if rows:
            # General zero-LLM aggregation fallback: return the deduped row count.
            return AnswerResult(
                answer=str(len(rows)),
                answer_type="count",
                confidence=0.55,
                evidence_ids=[r.evidence_id for r in rows if r.evidence_id][:20],
                reader_mode="heuristic_aggregation_count",
            )

    # For fact lookups, prefer the value field of the top item.
    for item in retrieved_items[:10]:
        mtype = item.get("memory_type", "")
        val = (item.get("value") or "").strip()
        if val and len(val) >= 3:
            extracted = _extract_first_person_answer(val)
            if extracted:
                ans, ans_type = extracted
                return AnswerResult(
                    answer=ans,
                    answer_type=ans_type,
                    confidence=float(item.get("confidence") or item.get("score") or 0.7),
                    evidence_ids=_evidence_ids([item]),
                    reader_mode="heuristic_first_person_np",
                )
            val = _strip_verbose_lead(val)
            return AnswerResult(
                answer=val,
                answer_type=_guess_answer_type(val),
                confidence=float(item.get("confidence") or item.get("score") or 0.6),
                evidence_ids=_evidence_ids([item]),
                reader_mode="heuristic_value",
            )

    # Fallback: first sentence of top text item
    top_text = (retrieved_items[0].get("text") or "").strip()
    top_text = _strip_verbose_lead(top_text)
    if top_text:
        extracted = _extract_first_person_answer(top_text)
        if extracted:
            ans, ans_type = extracted
            return AnswerResult(
                answer=ans,
                answer_type=ans_type,
                confidence=float(retrieved_items[0].get("score") or 0.55),
                evidence_ids=_evidence_ids(retrieved_items[:1]),
                reader_mode="heuristic_first_person_np",
            )
        first_sent = re.split(r"(?<=[.!?])\s+", top_text)[0][:240]
        return AnswerResult(
            answer=first_sent,
            answer_type=_guess_answer_type(first_sent),
            confidence=float(retrieved_items[0].get("score") or 0.4),
            evidence_ids=_evidence_ids(retrieved_items[:1]),
            reader_mode="heuristic_text",
        )

    return _ABSTAIN


def _guess_answer_type(text: str) -> AnswerType:
    t = text.strip().lower()
    if re.fullmatch(r"\d{4}[-/]\d{2}[-/]\d{2}.*", t) or re.fullmatch(r"\d{1,2}/\d{1,2}/\d{2,4}", t):
        return "date"
    if re.fullmatch(r"[-+]?\d[\d,.]*\s*(percent|%|items?|times?|days?|years?|hours?)?", t, re.I):
        return "count"
    if t in {"yes", "no", "true", "false"}:
        return "boolean"
    if "," in t and len(t.split(",")) >= 3:
        return "list"
    words = t.split()
    if len(words) == 1:
        return "entity" if text.strip()[0:1].isupper() else "noun_phrase"
    if len(words) <= 4:
        return "noun_phrase"
    return "noun_phrase"


# ---------------------------------------------------------------------------
# LLM reader prompts — one per strategy
# ---------------------------------------------------------------------------

_PROMPTS: dict[str, str] = {
    "fact_lookup": """\
Answer the question using ONLY the evidence items below.
Return a JSON object on a single line:
{{"answer": "<shortest correct answer>", "answer_type": "<entity|noun_phrase|date|count|boolean|list|explanation>", "confidence": 0.0-1.0, "evidence_ids": ["<id>", ...], "abstain": false}}

Rules:
- If answer is a name, return just the name. If a date, just the date. If a number, just the number.
- DATES: return the MOST SPECIFIC date the evidence supports. If the evidence shows a full
  day-month-year (e.g. "29 January, 2023"), return the full date — do NOT truncate to just the
  year or just the month. Only give a coarser date (year only, month+year) if that is genuinely
  all the evidence provides. Many evidence lines are timestamped as "[H:MM am/pm on D Month, YYYY]";
  treat that bracketed timestamp as the date of the event described in that line.
- Do NOT include sentences like "According to..." or "The context says...".
- If evidence is insufficient, set abstain=true and answer="unknown".

Question: {question}
Evidence:
{context}""",

    "aggregation": """\
Count or aggregate across ALL evidence items to answer the question.
Return a JSON object on a single line:
{{"answer": "<count or list>", "answer_type": "<count|list>", "confidence": 0.0-1.0, "evidence_ids": ["<id>", ...], "abstain": false}}

Rules:
- List every distinct matching item first (to avoid double-counting), then give the count/result.
- If evidence is insufficient, set abstain=true.

Question: {question}
Normalized evidence table ({n_items} rows after dedupe):
{context}""",

    "reasoning": """\
Reason across the evidence items to answer the multi-step question.
Return a JSON object on a single line:
{{"answer": "<concise answer>", "answer_type": "<noun_phrase|entity|explanation>", "confidence": 0.0-1.0, "evidence_ids": ["<id>", ...], "abstain": false}}

Rules:
- Be concise: no full explanations unless the question asks for one.
- If evidence is insufficient, set abstain=true.

Question: {question}
Evidence:
{context}""",

    "summary": """\
Summarize or answer the open-ended question using the evidence items.
Return a JSON object on a single line:
{{"answer": "<concise summary>", "answer_type": "explanation", "confidence": 0.0-1.0, "evidence_ids": ["<id>", ...], "abstain": false}}

Question: {question}
Evidence:
{context}""",
}


# ---------------------------------------------------------------------------
# LLM reader
# ---------------------------------------------------------------------------

LLM_READER_MODEL = os.environ.get("LLM_READER_MODEL", "")
_llm_client_cache: dict = {}


def _get_llm_client(model: str) -> dict | None:
    if model in _llm_client_cache:
        return _llm_client_cache[model]

    if not model:
        return None

    client_dict = None

    if model.startswith("openai/") or model.startswith("gpt"):
        try:
            from openai import OpenAI
            client_dict = {
                "provider": "openai",
                "client": OpenAI(),
                "model": model.removeprefix("openai/"),
            }
        except Exception as e:
            LOGGER.warning("OpenAI reader init failed: %s", e)

    elif model.startswith("anthropic/") or model.startswith("claude"):
        try:
            import anthropic
            client_dict = {
                "provider": "anthropic",
                "client": anthropic.Anthropic(),
                "model": model.removeprefix("anthropic/"),
            }
        except Exception as e:
            LOGGER.warning("Anthropic reader init failed: %s", e)

    elif model.startswith("google/") or model.startswith("gemini"):
        try:
            from google import genai
            client_dict = {
                "provider": "google",
                "client": genai.Client(),
                "model": model.removeprefix("google/"),
            }
        except Exception as e:
            LOGGER.warning("Google reader init failed: %s", e)

    if client_dict:
        _llm_client_cache[model] = client_dict
    return client_dict


def _call_llm(client_dict: dict, prompt: str, max_tokens: int = 300) -> str | None:
    try:
        provider = client_dict["provider"]
        if provider == "openai":
            resp = client_dict["client"].chat.completions.create(
                model=client_dict["model"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0,
            )
            return resp.choices[0].message.content.strip()
        elif provider == "anthropic":
            resp = client_dict["client"].messages.create(
                model=client_dict["model"],
                max_tokens=max_tokens,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()
        elif provider == "google":
            from google.genai import types
            resp = client_dict["client"].models.generate_content(
                model=client_dict["model"],
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=0,
                ),
            )
            return resp.text.strip()
    except Exception as e:
        LOGGER.warning("LLM reader call failed: %s", e)
        return None


_JSON_EXTRACT_RE = re.compile(r"\{.*\}", re.S)


def _parse_reader_response(raw: str) -> dict | None:
    if not raw:
        return None
    # Try straight parse
    try:
        return json.loads(raw)
    except Exception:
        pass
    # Try extracting the first {...} block
    m = _JSON_EXTRACT_RE.search(raw)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Answer postprocessor
# ---------------------------------------------------------------------------

_PROSE_LEAD_RE = re.compile(
    r"^(?:the answer is|based on(?: the| my)? (?:context|evidence|memory)|"
    r"according to(?: the)? (?:context|evidence|memory|retrieved)|"
    r"from the (?:context|evidence|memory))\s*[,:]?\s*",
    re.I,
)
_DATE_NORMALIZE_RE = re.compile(r"(\d{4})-0?(\d{1,2})-0?(\d{1,2})")


def postprocess_answer(answer: str, answer_type: AnswerType) -> str:
    """Normalize answer formatting to improve F1-style metrics."""
    answer = answer.strip()

    # Strip verbose prose leads (may need two passes for chained phrases)
    for _ in range(3):
        new = _PROSE_LEAD_RE.sub("", answer).strip()
        if new == answer:
            break
        answer = new

    # Strip enclosing quotes
    if len(answer) >= 2 and answer[0] == answer[-1] and answer[0] in '"\'':
        answer = answer[1:-1].strip()

    # For count/entity/noun_phrase: strip trailing sentence
    if answer_type in ("count", "entity", "noun_phrase"):
        # Take only the first sentence
        answer = re.split(r"(?<=[.!?])\s+", answer)[0]
        # Strip trailing punctuation
        answer = answer.rstrip(".,;:")

    # Normalize date to YYYY-MM-DD if date type and contains digits
    if answer_type == "date":
        answer = _DATE_NORMALIZE_RE.sub(r"\1-\2-\3", answer)

    return answer.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_answer(
    question: str,
    retrieved_items: list[dict],
    *,
    intent: str | None = None,
    model: str | None = None,
) -> AnswerResult:
    """Main entry point.

    Classifies intent, routes to the right strategy, and returns a structured
    AnswerResult.  Falls back to heuristic extraction when no LLM is available.
    """
    if not retrieved_items:
        return _ABSTAIN

    try:
        from common import classify_query_intent
    except ImportError:  # package-style import fallback
        from benchmarks.common import classify_query_intent
    resolved_intent = intent or classify_query_intent(question)
    strategy = _INTENT_TO_STRATEGY.get(resolved_intent, "fact_lookup")
    resolved_model = model or LLM_READER_MODEL
    client_dict = _get_llm_client(resolved_model) if resolved_model else None

    if client_dict is None:
        result = _heuristic_answer(question, retrieved_items, resolved_intent)
        result = AnswerResult(
            answer=postprocess_answer(result.answer, result.answer_type),
            answer_type=result.answer_type,
            confidence=result.confidence,
            evidence_ids=result.evidence_ids,
            abstain=result.abstain,
            reader_mode=result.reader_mode,
        )
        return result

    # Build prompt. Aggregation uses a normalized/deduped evidence table;
    # other strategies use compact evidence lines.
    if strategy == "aggregation":
        context, agg_rows = _build_aggregation_context(retrieved_items)
        n_items = len(agg_rows)
    else:
        context = _build_context(retrieved_items)
        n_items = len(retrieved_items)
    prompt_template = _PROMPTS.get(strategy, _PROMPTS["fact_lookup"])
    prompt = prompt_template.format(
        question=question,
        context=context,
        n_items=n_items,
    )

    raw = _call_llm(client_dict, prompt)
    parsed = _parse_reader_response(raw) if raw else None

    if not parsed:
        LOGGER.warning("Reader parse failure, falling back to heuristic (raw=%r)", (raw or "")[:200])
        return _heuristic_answer(question, retrieved_items, resolved_intent)

    answer = postprocess_answer(
        str(parsed.get("answer", "unknown") or "unknown"),
        parsed.get("answer_type", "noun_phrase"),
    )
    return AnswerResult(
        answer=answer,
        answer_type=parsed.get("answer_type", "noun_phrase"),
        confidence=float(parsed.get("confidence", 0.7)),
        evidence_ids=parsed.get("evidence_ids") or _evidence_ids(retrieved_items[:5]),
        abstain=bool(parsed.get("abstain", False)) or answer.lower() == "unknown",
        reader_mode=f"llm_{strategy}",
    )


def read_answer_str(
    question: str,
    retrieved_items: list[dict],
    *,
    intent: str | None = None,
    model: str | None = None,
) -> str | None:
    """Convenience wrapper: return just the answer string or None on abstain.

    Intended as a drop-in replacement for the legacy llm_read_answer()
    in common.py benchmarks.
    """
    result = read_answer(question, retrieved_items, intent=intent, model=model)
    if result.abstain:
        return None
    return result.answer
