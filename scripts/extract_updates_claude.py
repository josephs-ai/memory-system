"""
Optional Claude (Haiku) memory extractor.

A second opt-in extraction mode alongside the Qwen/Ollama one. The default
pipeline stays the deterministic structural extractor; this is only used when
something asks for it explicitly.

Everything except the transport is imported from extract_updates_qwen: the
prompt, the JSON coercion, the schema validation, the legacy-item shaping. Two
extractors that disagree about what a valid candidate looks like would be worse
than having one, and the prompt is the part most likely to drift.

Billing note: this runs on the Anthropic API key in openclaw's config, not on a
Claude subscription. A `claude setup-token` credential is issued for Claude Code
and Anthropic rejects it from other clients (HTTP 401, "OAuth access token is
invalid"), so subscription-funded extraction is not available here.

Environment knobs:
  ANTHROPIC_API_KEY            overrides the key read from openclaw's config
  OPENCLAW_CLAUDE_MODEL        default: claude-haiku-4-5
  OPENCLAW_CLAUDE_TIMEOUT      default: 120
  OPENCLAW_CLAUDE_MAX_TOKENS   default: 4096
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib import error, request

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from extract_updates_qwen import (  # noqa: E402  (path set above)
    PROMPT_TEMPLATE,
    normalize_item,
    parse_candidates,
    strip_json_noise,
)

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
DEFAULT_MODEL = os.environ.get("OPENCLAW_CLAUDE_MODEL", "claude-haiku-4-5")
DEFAULT_TIMEOUT = float(os.environ.get("OPENCLAW_CLAUDE_TIMEOUT", "120"))
DEFAULT_MAX_TOKENS = int(os.environ.get("OPENCLAW_CLAUDE_MAX_TOKENS", "4096"))

OPENCLAW_CONFIG = Path.home() / ".openclaw" / "openclaw.json"


def resolve_api_key() -> str:
    """Env first, then the anthropic provider in openclaw's config."""
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key
    try:
        cfg = json.loads(OPENCLAW_CONFIG.read_text())
        key = (cfg.get("models", {}).get("providers", {})
                  .get("anthropic", {}).get("apiKey", "") or "").strip()
    except Exception:
        key = ""
    if not key:
        raise SystemExit(
            "no Anthropic API key: set ANTHROPIC_API_KEY or "
            f"models.providers.anthropic.apiKey in {OPENCLAW_CONFIG}"
        )
    return key


def claude_generate(prompt: str, *, model: str, timeout: float, max_tokens: int) -> str:
    body = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        # Deterministic-ish: this is extraction, not writing.
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()

    req = request.Request(API_URL, data=body, method="POST", headers={
        "content-type": "application/json",
        "x-api-key": resolve_api_key(),
        "anthropic-version": API_VERSION,
    })
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode()[:300]
        except Exception:
            pass
        raise SystemExit(f"anthropic HTTP {exc.code}: {detail or exc.reason}")
    except error.URLError as exc:
        raise SystemExit(f"anthropic unreachable: {exc.reason}")

    # Usage goes to stderr so stdout stays pure JSONL for the routing step.
    usage = payload.get("usage") or {}
    print(json.dumps({"claude_usage": {
        "model": model,
        "input": usage.get("input_tokens"),
        "output": usage.get("output_tokens"),
    }}), file=sys.stderr)

    parts = [b.get("text", "") for b in payload.get("content", []) if b.get("type") == "text"]
    return "".join(parts)


def extract_with_claude(
    chunk_text: str,
    *,
    source_agent: str,
    source_session: str,
    source_chunk: str,
    model: str,
    timeout: float,
    max_tokens: int,
) -> list[dict]:
    # Same template and same 24k truncation as the Qwen path, so the two modes
    # cannot drift apart on what they ask for.
    prompt = PROMPT_TEMPLATE.format(
        source_agent=source_agent,
        source_session=source_session,
        source_chunk=source_chunk,
        text=chunk_text[:24000],
    )
    raw = claude_generate(prompt, model=model, timeout=timeout, max_tokens=max_tokens)

    items = []
    for idx, cand in enumerate(parse_candidates(strip_json_noise(raw))):
        candidate = normalize_item(
            cand,
            source_agent=source_agent,
            source_session=source_session,
            source_chunk=source_chunk,
            index=idx,
            extractor="claude",
            extractor_model=model,
            id_prefix="claude_",
        )
        if candidate is not None:
            items.append(candidate.to_legacy_item())
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract memory candidates with Claude.")
    ap.add_argument("chunk_file")
    ap.add_argument("--source-agent", default="unknown")
    ap.add_argument("--source-session", default="unknown")
    ap.add_argument("--source-chunk", default="")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    args = ap.parse_args()

    path = Path(args.chunk_file)
    if not path.exists():
        raise SystemExit(f"no such chunk: {path}")

    items = extract_with_claude(
        path.read_text(encoding="utf-8", errors="ignore"),
        source_agent=args.source_agent,
        source_session=args.source_session,
        source_chunk=args.source_chunk or path.name,
        model=args.model,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
    )
    for item in items:
        print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    main()
