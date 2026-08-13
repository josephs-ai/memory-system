"""
Agent-facing entry point to the Memory Engine.

AGENTS.md has long instructed agents to "read the past 3 days of memory via the
Memory Engine" without naming a command, so the instruction was unactionable.
This is that command.

    engram.py "watermark sync"          semantic search: what do we know
    engram.py --since 1d                episodic: what happened recently
    engram.py --since 3d "phase 4"      that term, within that window

Two lanes, on purpose. memory_items answers "what is true" and is deduped by
content, so asking it "what did we do yesterday" returns whatever old rows share
vocabulary. memory_episodes answers "what happened, when" and is never deduped,
so a time window is a filter there rather than a ranking hint.

Prefers the search service on :8791 (vector + graph + reranking) and falls back
to a direct database query when it is down, because an agent mid-task needs an
answer more than it needs the best possible ranking.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

SERVICE = "http://127.0.0.1:8791"
_UNITS = {"h": "hours", "d": "days", "w": "weeks"}


def parse_since(value: str) -> datetime:
    now = datetime.now(timezone.utc)
    text = (value or "").strip().lower()
    if text in {"today", "day"}:
        return now - timedelta(days=1)
    if text == "yesterday":
        return now - timedelta(days=2)
    if text == "week":
        return now - timedelta(weeks=1)
    m = re.fullmatch(r"(\d+)\s*([hdw])", text)
    if m:
        return now - timedelta(**{_UNITS[m.group(2)]: int(m.group(1))})
    try:
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        raise SystemExit(f"bad --since {value!r} (try: 1d, yesterday, 2w, 2026-08-01)")


def search_via_service(query: str, limit: int, timeout: int):
    params = urllib.parse.urlencode({"query": query, "top_k": limit})
    with urllib.request.urlopen(f"{SERVICE}/search?{params}", timeout=timeout) as resp:
        return json.loads(resp.read()).get("results", [])


def search_via_db(query: str, limit: int):
    """Fallback: same ranking function the service uses, minus graph and rerank."""
    from sentence_transformers import SentenceTransformer
    from memory_db import hybrid_search_memory_items

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    return hybrid_search_memory_items(
        model.encode(query).tolist(), query_text=query, limit=limit
    )


def main():
    parser = argparse.ArgumentParser(description="Query the Memory Engine.")
    parser.add_argument("query", nargs="*", help="what to look for")
    parser.add_argument("--since", default=None, help="1d, yesterday, 2w, or ISO date")
    parser.add_argument("--until", default=None)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--agent", default=None, help="episodes: restrict to one agent")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    query = " ".join(args.query).strip()

    # A time window means the question is about activity, so answer from the
    # episodic lane -- that is the whole reason it exists.
    if args.since:
        from memory_db import search_episodes, close_pool

        after = parse_since(args.since)
        before = parse_since(args.until) if args.until else None
        rows = search_episodes(
            after_ts=after, before_ts=before, query_text=query or None,
            agent=args.agent, limit=args.limit,
        )
        label = f'"{query}" ' if query else ""
        print(f"{len(rows)} episode(s) {label}since {after:%Y-%m-%d %H:%M}")
        for e in rows:
            print(f"  {e['started_at']:%Y-%m-%d %H:%M} [{e['agent']}] {e['summary'][:110]}")
        if not rows:
            any_rows = search_episodes(after_ts=after, before_ts=before, limit=1)
            print("  (episodes exist in this window; none matched the text)"
                  if query and any_rows else "  (no activity recorded in this window)")
        close_pool()
        return

    if not query:
        parser.error("give a query, or --since for a time window")

    try:
        results = search_via_service(query, args.limit, args.timeout)
        source = "service"
    except Exception as exc:
        print(f"[service unavailable: {type(exc).__name__}; using direct db]", file=sys.stderr)
        results = search_via_db(query, args.limit)
        source = "db"

    print(f"{len(results)} result(s) for \"{query}\" [{source}]")
    for r in results:
        text = " ".join(str(r.get("text") or "").split())
        meta = r.get("memory_type") or r.get("source_type") or ""
        print(f"  [{meta}] {text[:150]}")

    if source == "db":
        from memory_db import close_pool
        close_pool()


if __name__ == "__main__":
    main()
