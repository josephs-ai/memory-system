"""
Extract chunk updates from source data.

Key functions: run_stage, load_jsonl_text, dedupe_items, run_structural_extract
"""
import json
import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

WORKSPACE = Path.home() / ".openclaw" / "workspace"
SCRIPTS_DIR = WORKSPACE / ".memory-index" / "scripts"

EXTRACT_SCRIPT = SCRIPTS_DIR / "extract_updates.py"
GENERATE_SCRIPT = SCRIPTS_DIR / "generate_memory_candidates.py"
FILTER_SCRIPT = SCRIPTS_DIR / "filter_memory_candidates.py"
JUDGE_SCRIPT = SCRIPTS_DIR / "judge_memory_candidates.py"
CACHE_FILENAME = ".extract_chunk_cache.json"
CACHE_FORMAT_VERSION = 3
EXTRACTOR_FINGERPRINT = f"extract_chunk_updates:v{CACHE_FORMAT_VERSION}"
CACHE_MAX_ENTRIES = 1000


def run_stage(cmd):
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()



def load_jsonl_text(text: str):
    items = []
    if not text or text.strip() == "NONE":
        return items
    for line in text.splitlines():
        s = line.strip()
        if not s or s == "NONE":
            continue
        try:
            items.append(json.loads(s))
        except Exception:
            pass
    return items



def dedupe_items(items):
    seen = set()
    out = []
    for item in items:
        key = (
            item.get("claim_type") or item.get("memory_type"),
            json.dumps(item.get("scope_envelope") or {}, sort_keys=True),
            item.get("normalized_text") or item.get("text") or item.get("raw_text"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out



def load_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {}
    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    if raw.get("cache_format_version") != CACHE_FORMAT_VERSION:
        return {}
    entries = raw.get("entries")
    return entries if isinstance(entries, dict) else {}



def save_cache(cache_path: Path, cache: dict):
    payload = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "extractor_fingerprint": EXTRACTOR_FINGERPRINT,
        "entries": cache,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=cache_path.name + ".", suffix=".tmp", dir=str(cache_path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, cache_path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass



def chunk_signature(chunk_file: Path) -> dict:
    text = chunk_file.read_text(encoding="utf-8", errors="ignore")
    return {
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "size": len(text),
    }



def load_cursor_file(cursor_file: str | None) -> dict | None:
    if not cursor_file:
        return None
    try:
        raw = json.loads(Path(cursor_file).read_text(encoding="utf-8"))
    except Exception:
        return None
    return raw if isinstance(raw, dict) else None



def stable_session_family(source_session: str, cursor_data: dict | None = None) -> str:
    # Prefer stable transcript identity from the committed cursor when present.
    try:
        if cursor_data:
            head_hash = cursor_data.get("head_hash")
            inode = cursor_data.get("inode")
            if isinstance(head_hash, str) and head_hash and inode is not None:
                return f"{head_hash[:12]}-{inode}"
    except Exception:
        pass

    # Collapse volatile filenames like UUID/topic session names into a stable family when possible.
    return Path(source_session).stem.split(".")[0]



def cache_key(
    chunk_file: Path,
    *,
    source_agent: str,
    source_session: str,
    legacy_pipeline: bool,
    extractor_fingerprint: str,
    cursor_data: dict | None = None,
) -> str:
    return json.dumps(
        {
            "chunk": chunk_file.name,
            "source_agent": source_agent,
            "session_family": stable_session_family(source_session, cursor_data),
            "legacy_pipeline": legacy_pipeline,
            "extractor_fingerprint": extractor_fingerprint,
        },
        sort_keys=True,
    )



def get_cached_items(cache: dict, key: str, sig: dict) -> list[dict] | None:
    entry = cache.get(key)
    if not entry:
        return None
    if entry.get("sha256") != sig["sha256"] or entry.get("size") != sig["size"]:
        return None
    items = entry.get("items")
    if not isinstance(items, list):
        return None
    entry["last_touched_ms"] = int(time.time() * 1000)
    return items



def prune_cache(cache: dict):
    if len(cache) <= CACHE_MAX_ENTRIES:
        return
    victims = sorted(
        cache.items(),
        key=lambda kv: kv[1].get("last_touched_ms", kv[1].get("created_ms", 0)),
    )
    for key, _ in victims[: max(0, len(cache) - CACHE_MAX_ENTRIES)]:
        cache.pop(key, None)



def put_cached_items(cache: dict, key: str, sig: dict, items: list[dict]):
    now_ms = int(time.time() * 1000)
    cache[key] = {
        "sha256": sig["sha256"],
        "size": sig["size"],
        "items": items,
        "created_ms": now_ms,
        "last_touched_ms": now_ms,
    }
    prune_cache(cache)



def run_structural_extract(chunk_file: Path, *, source_agent: str, source_session: str) -> list[dict]:
    extracted = run_stage(
        [
            "python3",
            str(EXTRACT_SCRIPT),
            str(chunk_file),
            "--source-agent",
            source_agent,
            "--source-session",
            source_session,
            "--source-chunk",
            chunk_file.name,
        ]
    )
    return load_jsonl_text(extracted)



def run_legacy_extract(chunk_file: Path, *, source_agent: str, source_session: str) -> list[dict]:
    generated = run_stage([
        "python3",
        str(GENERATE_SCRIPT),
        str(chunk_file),
        "--source-agent", source_agent,
        "--source-session", source_session,
        "--source-chunk", chunk_file.name,
    ])
    if not generated or generated == "NONE":
        return []

    import tempfile

    generated_tmp = Path(tempfile.mkstemp(suffix=".generated.jsonl")[1])
    generated_tmp.write_text(generated + ("\n" if generated and not generated.endswith("\n") else ""), encoding="utf-8")
    try:
        filtered = run_stage(["python3", str(FILTER_SCRIPT), str(generated_tmp)])
        if not filtered or filtered == "NONE":
            return []
        filtered_tmp = Path(tempfile.mkstemp(suffix=".filtered.jsonl")[1])
        filtered_tmp.write_text(filtered + ("\n" if filtered and not filtered.endswith("\n") else ""), encoding="utf-8")
        try:
            judged = run_stage(["python3", str(JUDGE_SCRIPT), str(filtered_tmp)])
            return load_jsonl_text(judged)
        finally:
            try:
                filtered_tmp.unlink()
            except Exception:
                pass
    finally:
        try:
            generated_tmp.unlink()
        except Exception:
            pass



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-dir", required=True)
    parser.add_argument("--source-agent", default="unknown")
    parser.add_argument("--source-session", default="unknown")
    parser.add_argument("--reduced-only", action="store_true")
    parser.add_argument("--legacy-pipeline", action="store_true")
    parser.add_argument("--shadow-compare", action="store_true")
    parser.add_argument("--ignore-cache", action="store_true")
    parser.add_argument("--cursor-file")
    args = parser.parse_args()

    chunk_dir = Path(args.chunk_dir)
    cursor_data = load_cursor_file(args.cursor_file)
    chunk_files = sorted(chunk_dir.glob("*.txt"))
    cache_path = chunk_dir / CACHE_FILENAME
    cache = load_cache(cache_path)
    cache_dirty = False

    if not chunk_files:
        print("No chunk files found.")
        return

    reduced = []

    for chunk_file in chunk_files:
        if not args.reduced_only:
            print(f"===== {chunk_file.name} =====")

        sig = chunk_signature(chunk_file)
        key = cache_key(
            chunk_file,
            source_agent=args.source_agent,
            source_session=args.source_session,
            legacy_pipeline=args.legacy_pipeline,
            extractor_fingerprint=EXTRACTOR_FINGERPRINT,
            cursor_data=cursor_data,
        )
        cached_items = None if args.ignore_cache else get_cached_items(cache, key, sig)

        if cached_items is not None:
            items = cached_items
            cache_dirty = True
            print(
                json.dumps({"chunk": chunk_file.name, "cache": "hit", "items": len(items)}),
                file=sys.stderr,
            )
        else:
            if args.legacy_pipeline:
                items = run_legacy_extract(
                    chunk_file,
                    source_agent=args.source_agent,
                    source_session=args.source_session,
                )
            else:
                items = run_structural_extract(
                    chunk_file,
                    source_agent=args.source_agent,
                    source_session=args.source_session,
                )
                if args.shadow_compare:
                    legacy_items = run_legacy_extract(
                        chunk_file,
                        source_agent=args.source_agent,
                        source_session=args.source_session,
                    )
                    print(
                        json.dumps(
                            {
                                "chunk": chunk_file.name,
                                "structural_count": len(items),
                                "legacy_count": len(legacy_items),
                            }
                        ),
                        file=sys.stderr,
                    )
            items = dedupe_items(items)
            put_cached_items(cache, key, sig, items)
            cache_dirty = True
            print(
                json.dumps({"chunk": chunk_file.name, "cache": "miss", "items": len(items)}),
                file=sys.stderr,
            )

        items = dedupe_items(items)

        if not items:
            if not args.reduced_only:
                print("NONE")
                print()
        else:
            if not args.reduced_only:
                for item in items:
                    print(json.dumps(item, ensure_ascii=False))
                print()
            reduced.extend(items)

    reduced = dedupe_items(reduced)

    if cache_dirty:
        save_cache(cache_path, cache)

    if args.reduced_only:
        if not reduced:
            print("NONE")
        else:
            for item in reduced:
                print(json.dumps(item, ensure_ascii=False))
    else:
        print("===== REDUCED UPDATES =====")
        if not reduced:
            print("NONE")
        else:
            for item in reduced:
                print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    main()
