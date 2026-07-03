"""
Optional Qwen/Ollama memory extractor.

This is an opt-in second extraction mode for the memory system. The default
pipeline remains the deterministic structural extractor. This script reads a
transcript chunk, asks a local/remote Ollama Qwen model for durable memory
candidates, validates them with the normal memory schema, and prints legacy
JSONL items compatible with route_memory_items_batch.py.

Environment knobs:
  OPENCLAW_QWEN_OLLAMA_URL   default: http://127.0.0.1:11434
  OPENCLAW_QWEN_MODEL        default: qwen3:32b
  OPENCLAW_QWEN_TIMEOUT      default: 180
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib import request, error

from memory_candidate_schema import (
    AuthorityBasis,
    ClaimType,
    DurabilityClass,
    ImpactLevel,
    ScopeEnvelope,
    StructuredMemoryCandidate,
)
from memory_validation_gate import validate_candidate

DEFAULT_OLLAMA_URL = os.environ.get("OPENCLAW_QWEN_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
DEFAULT_MODEL = os.environ.get("OPENCLAW_QWEN_MODEL", "qwen3:32b")
DEFAULT_TIMEOUT = float(os.environ.get("OPENCLAW_QWEN_TIMEOUT", "180"))

ALLOWED_CLAIM_TYPES = {item.value for item in ClaimType}
ALLOWED_AUTHORITY = {item.value for item in AuthorityBasis}
ALLOWED_DURABILITY = {item.value for item in DurabilityClass}
ALLOWED_IMPACT = {item.value for item in ImpactLevel}

PROMPT_TEMPLATE = """You extract durable long-term memory candidates from OpenClaw transcript chunks.

Return ONLY valid JSON. Do not include markdown. Do not include reasoning.
The output must be a JSON array. Each array item must have:
- claim_text: one atomic durable claim, decision, user preference, rule, bug history, or implementation pattern.
- claim_type: one of rule, decision, preference, fact, implementation_pattern, bug_history, open_question, summary_only.
- authority_basis: one of user_explicit, developer_explicit, tester_explicit, system_tool_verified, assistant_inferred, summary_inferred, unknown.
- durability_class: one of ephemeral, session, candidate, review_required, durable_eligible.
- impact_level: one of low, medium, high, critical.
- subject: short optional subject/entity string.
- predicate: short optional predicate/property string.
- object: short optional object/value string.
- scope_type: one of user, project, system, workflow, agent, unknown.
- scope_id: concise optional id, e.g. memory-system, openclaw, user-profile.
- extractor_confidence: number from 0.0 to 1.0.

Strict rules:
- Extract only durable information that would be useful in future sessions.
- Ignore transient chatter, one-off status messages, tool logs, and repeated text.
- Do not store private secrets, API keys, tokens, passwords, or raw credentials.
- Do not store chain-of-thought or hidden reasoning.
- Prefer explicit user/developer/tool-verified facts over assistant guesses.
- Keep each claim under 35 words when possible.
- If there are no durable candidates, return [].

Source metadata:
source_agent={source_agent}
source_session={source_session}
source_chunk={source_chunk}

Transcript chunk:
---
{text}
---
"""


def stable_candidate_id(*parts: str) -> str:
    payload = "\u241f".join(part or "" for part in parts)
    return "qwen_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def strip_json_noise(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    # Qwen sometimes wraps JSON in prose despite instructions. Prefer the first array.
    if not text.startswith("["):
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            text = text[start:end + 1]
    return text.strip()


def ollama_generate(prompt: str, *, ollama_url: str, model: str, timeout: float) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_ctx": 8192,
        },
    }
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{ollama_url}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except error.URLError as exc:
        raise RuntimeError(f"ollama request failed url={ollama_url!r} model={model!r}: {exc}") from exc
    response = data.get("response", "")
    if not isinstance(response, str):
        raise RuntimeError("ollama response did not contain a string 'response' field")
    return response


def coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return str(value).strip() or None


def coerce_choice(value: Any, allowed: set[str], default: str) -> str:
    value = coerce_str(value)
    if not value:
        return default
    value = value.lower().strip().replace(" ", "_")
    return value if value in allowed else default


def coerce_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except Exception:
        confidence = 0.5
    return max(0.0, min(1.0, confidence))


def parse_candidates(raw: str) -> list[dict[str, Any]]:
    cleaned = strip_json_noise(raw)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Qwen did not return parseable JSON: {exc}: {cleaned[:500]}") from exc
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        payload = payload["items"]
    if not isinstance(payload, list):
        raise RuntimeError("Qwen output was not a JSON array")
    return [item for item in payload if isinstance(item, dict)]


def normalize_item(
    item: dict[str, Any],
    *,
    source_agent: str,
    source_session: str,
    source_chunk: str,
    index: int,
) -> StructuredMemoryCandidate | None:
    claim_text = coerce_str(item.get("claim_text") or item.get("text") or item.get("raw_text"))
    if not claim_text:
        return None
    lowered = claim_text.lower()
    if any(secret in lowered for secret in ("api key", "secret key", "password", "token:", "sk-", "bearer ")):
        return None

    claim_type = coerce_choice(item.get("claim_type"), ALLOWED_CLAIM_TYPES, ClaimType.fact.value)
    authority = coerce_choice(item.get("authority_basis"), ALLOWED_AUTHORITY, AuthorityBasis.unknown.value)
    durability = coerce_choice(item.get("durability_class"), ALLOWED_DURABILITY, DurabilityClass.review_required.value)
    impact = coerce_choice(item.get("impact_level"), ALLOWED_IMPACT, ImpactLevel.low.value)

    scope_type = coerce_str(item.get("scope_type")) or "unknown"
    scope_id = coerce_str(item.get("scope_id"))
    scope = ScopeEnvelope(
        scope_id=scope_id,
        scope_type=scope_type,
        scope_confidence=coerce_confidence(item.get("scope_confidence", item.get("extractor_confidence", 0.5))),
        raw={"extractor": "qwen", "model": DEFAULT_MODEL},
    )

    candidate = StructuredMemoryCandidate(
        candidate_id=stable_candidate_id(source_agent, source_session, source_chunk, str(index), claim_text),
        source_agent=source_agent,
        source_session=source_session,
        source_chunk=source_chunk,
        claim_text=claim_text,
        claim_type=ClaimType(claim_type),
        subject=coerce_str(item.get("subject")),
        predicate=coerce_str(item.get("predicate")),
        object=coerce_str(item.get("object")),
        scope_envelope=scope,
        authority_basis=AuthorityBasis(authority),
        durability_class=DurabilityClass(durability),
        impact_level=ImpactLevel(impact),
        requires_approval=True,
        extractor_confidence=coerce_confidence(item.get("extractor_confidence")),
        normalization_notes=["extracted_by_qwen_ollama_optional_mode"],
    )
    return validate_candidate(candidate)


def extract_with_qwen(
    text: str,
    *,
    source_agent: str,
    source_session: str,
    source_chunk: str,
    ollama_url: str,
    model: str,
    timeout: float,
) -> list[dict[str, Any]]:
    prompt = PROMPT_TEMPLATE.format(
        source_agent=source_agent,
        source_session=source_session,
        source_chunk=source_chunk,
        text=text[:24000],
    )
    raw = ollama_generate(prompt, ollama_url=ollama_url, model=model, timeout=timeout)
    candidates = []
    for idx, item in enumerate(parse_candidates(raw)):
        candidate = normalize_item(
            item,
            source_agent=source_agent,
            source_session=source_session,
            source_chunk=source_chunk,
            index=idx,
        )
        if candidate is not None:
            candidates.append(candidate.to_legacy_item())
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("--source-agent", default="unknown")
    parser.add_argument("--source-session", default="unknown")
    parser.add_argument("--source-chunk", default=None)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    args = parser.parse_args()

    path = Path(args.input_path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    try:
        items = extract_with_qwen(
            text,
            source_agent=args.source_agent,
            source_session=args.source_session,
            source_chunk=args.source_chunk or path.name,
            ollama_url=args.ollama_url,
            model=args.model,
            timeout=args.timeout,
        )
    except Exception as exc:
        print(f"QWEN_EXTRACT_ERROR {exc}", file=sys.stderr)
        raise SystemExit(2)

    if not items:
        print("NONE")
        return
    for item in items:
        print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    main()
