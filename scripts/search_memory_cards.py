#!/usr/bin/env python3
"""
Memory system utility: search the READABLE memory layer (cards + digests).

Phase 5 (card search). Searches the index built by card_search_index.py and
returns the session card / daily roll-up / digest node that best answers a query,
with a snippet + path + provenance.

Why a lexical-first scorer: the existing DB-vector hybrid stack
(retrieve_memory_hybrid / memory_db.hybrid_search_memory_items) covers raw
memory_items but (a) requires the embedding + search service to be up, and (b)
does NOT search the human-readable cards/digests. This searcher is pure-stdlib so
the readable layer stays queryable EVEN WHEN that service is DOWN — the recurring
failure this upgrade hardens against. When the service IS healthy, callers may
blend in semantic hits (semantic_available hook); when not, we fall back to
lexical and SAY so (degraded, never silent).

Scoring (deterministic):
  score = sum_over_query_terms( tf_in_field * field_weight )
          + recency_boost(generated_at/day)
          + kind_boost (digest > daily > session: compounded knowledge ranks up)
          - freshness_penalty (stale/degraded down-weighted)
Tie-break: (-score, freshness_rank, generated_at desc, id) for stable output.

Key functions: search_cards, score_record, tokenize.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import card_search_index as csi  # noqa: E402

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "was", "were", "be", "we", "i", "it", "that", "this", "with", "what", "why",
    "how", "do", "did", "does", "about", "from", "at", "as", "by",
}

_FIELD_WEIGHT_TITLE = 3.0
_FIELD_WEIGHT_SLUG = 4.0
_FIELD_WEIGHT_BODY = 1.0
_KIND_BOOST = {"digest": 1.5, "daily": 0.5, "session": 0.0}
_FRESHNESS_RANK = {"fresh": 0, "unknown": 1, "stale": 2, "degraded": 3}
_FRESHNESS_PENALTY = {"fresh": 0.0, "unknown": 0.0, "stale": 1.5, "degraded": 3.0}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOPWORDS]


def _term_freq(tokens: list[str]) -> dict[str, int]:
    tf: dict[str, int] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    return tf


def _recency_boost(generated_at: str, day: str | None) -> float:
    """Newer cards rank slightly higher. Bounded, log-decayed by age in days."""
    stamp = generated_at or (f"{day}T00:00:00Z" if day else "")
    if not stamp:
        return 0.0
    try:
        t = time.strptime(stamp[:19], "%Y-%m-%dT%H:%M:%S")
        age_days = max(0.0, (time.time() - time.mktime(t)) / 86400.0)
    except (ValueError, OverflowError):
        return 0.0
    # ~+1.0 today, decaying toward 0 over weeks
    return 1.0 / (1.0 + math.log1p(age_days))


@dataclass
class SearchHit:
    record: csi.CardRecord
    score: float
    snippet: str

    def to_dict(self) -> dict:
        r = self.record
        return {
            "id": r.id,
            "kind": r.kind,
            "score": round(self.score, 4),
            "title": r.title,
            "path": r.path,
            "freshness": r.freshness,
            "generated_at": r.generated_at,
            "agent": r.agent,
            "day": r.day,
            "digest_kind": r.digest_kind,
            "slug": r.slug,
            "snippet": self.snippet,
        }


def _make_snippet(record: csi.CardRecord, query_tokens: set[str], width: int = 200) -> str:
    """Return the first body line that hits a query term, else the first line."""
    lines = [l for l in record.text.splitlines() if l.strip()]
    for line in lines:
        if query_tokens & set(tokenize(line)):
            return line[:width]
    return (lines[0][:width] if lines else record.title[:width])


def score_record(record: csi.CardRecord, query_tf: dict[str, int]) -> float:
    title_tf = _term_freq(tokenize(record.title))
    slug_tf = _term_freq(tokenize(record.slug or ""))
    body_tf = _term_freq(tokenize(record.text))

    raw = 0.0
    for term, q in query_tf.items():
        hit = (
            title_tf.get(term, 0) * _FIELD_WEIGHT_TITLE
            + slug_tf.get(term, 0) * _FIELD_WEIGHT_SLUG
            + body_tf.get(term, 0) * _FIELD_WEIGHT_BODY
        )
        if hit:
            # sublinear in tf, scaled by query term frequency
            raw += q * (1.0 + math.log1p(hit))
    if raw <= 0.0:
        return 0.0
    raw += _KIND_BOOST.get(record.kind, 0.0)
    raw += _recency_boost(record.generated_at, record.day)
    raw -= _FRESHNESS_PENALTY.get(record.freshness, 0.0)
    return raw


def search_cards(
    query: str,
    *,
    top_k: int = 8,
    kind: str | None = None,
    agent: str | None = None,
    digest_kind: str | None = None,
    records: list[csi.CardRecord] | None = None,
    semantic_blend=None,
) -> tuple[list[SearchHit], dict]:
    """Search the readable card index.

    Returns (hits, diagnostics). diagnostics carries degraded/lexical-only state
    so callers (and tests) can SEE when the semantic blend was unavailable.

    semantic_blend: optional callable(query, records) -> {id: bonus_score}. Used
    only when the search service is healthy; any exception or None -> lexical-only,
    flagged degraded in diagnostics.
    """
    if records is None:
        _meta, records = csi.load_index()

    diag: dict = {
        "index_records": len(records),
        "mode": "lexical",
        "degraded": False,
        "notes": [],
    }
    if not records:
        diag["degraded"] = True
        diag["notes"].append("card index empty or missing — run card_search_index.py")
        return [], diag

    query_tf = _term_freq(tokenize(query))
    if not query_tf:
        diag["notes"].append("query had no searchable terms after stopword removal")
        return [], diag

    # filter
    def keep(r: csi.CardRecord) -> bool:
        if kind and r.kind != kind:
            return False
        if agent and r.agent != agent:
            return False
        if digest_kind and r.digest_kind != digest_kind:
            return False
        return True

    pool = [r for r in records if keep(r)]

    semantic_bonus: dict[str, float] = {}
    if semantic_blend is not None:
        try:
            semantic_bonus = semantic_blend(query, pool) or {}
            diag["mode"] = "hybrid"
        except Exception as exc:  # noqa: BLE001 — degrade, never crash search
            diag["degraded"] = True
            diag["notes"].append(f"semantic blend unavailable, lexical-only: {exc}")
    else:
        diag["notes"].append("semantic blend not requested (lexical-only)")

    qtok = set(query_tf)
    scored: list[SearchHit] = []
    for r in pool:
        s = score_record(r, query_tf) + float(semantic_bonus.get(r.id, 0.0))
        if s > 0.0:
            scored.append(SearchHit(record=r, score=s, snippet=_make_snippet(r, qtok)))

    scored.sort(key=lambda h: (
        -h.score,
        _FRESHNESS_RANK.get(h.record.freshness, 9),
        # newer first within ties -> invert lexicographic stamp via reverse later
        h.record.generated_at,
        h.record.id,
    ))
    # apply generated_at-desc within identical score+freshness without disturbing
    # the primary order: stable sort by (-score, freshness) already done; re-stable
    # by generated_at desc as a secondary pass keeps determinism.
    scored.sort(key=lambda h: (-h.score, _FRESHNESS_RANK.get(h.record.freshness, 9)))
    diag["matched"] = len(scored)
    return scored[:top_k], diag


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Search readable memory cards/digests.")
    ap.add_argument("--query", required=True)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--kind", choices=["session", "daily", "digest"])
    ap.add_argument("--agent")
    ap.add_argument("--digest-kind")
    ap.add_argument("--rebuild-index", action="store_true",
                    help="rebuild card_index.jsonl before searching")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.rebuild_index:
        csi.build_card_index()

    hits, diag = search_cards(
        args.query,
        top_k=args.top_k,
        kind=args.kind,
        agent=args.agent,
        digest_kind=args.digest_kind,
    )

    if args.json:
        print(json.dumps(
            {"diagnostics": diag, "hits": [h.to_dict() for h in hits]},
            ensure_ascii=False, sort_keys=True))
        return 0

    if diag.get("degraded"):
        print(f"[degraded] {'; '.join(diag['notes'])}", file=sys.stderr)
    if not hits:
        print("no matching cards.")
        return 0
    for i, h in enumerate(hits, 1):
        loc = h.record.digest_kind and f"digest/{h.record.digest_kind}/{h.record.slug}" \
            or (h.record.kind == "daily" and f"daily/{h.record.day}/{h.record.agent}") \
            or h.record.id
        print(f"{i:2}. [{h.score:5.2f}] {h.record.kind:7} {loc}  ({h.record.freshness})")
        print(f"      {h.snippet}")
        print(f"      -> {h.record.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
