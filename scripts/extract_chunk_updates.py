"""
Extract chunk updates from source data.

Key functions: run_stage, load_jsonl_text, dedupe_items, run_structural_extract
"""
import json
import argparse
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path.home() / ".openclaw" / "workspace"
SCRIPTS_DIR = WORKSPACE / ".memory-index" / "scripts"

EXTRACT_SCRIPT = SCRIPTS_DIR / "extract_updates.py"
GENERATE_SCRIPT = SCRIPTS_DIR / "generate_memory_candidates.py"
FILTER_SCRIPT = SCRIPTS_DIR / "filter_memory_candidates.py"
JUDGE_SCRIPT = SCRIPTS_DIR / "judge_memory_candidates.py"


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
    args = parser.parse_args()

    chunk_dir = Path(args.chunk_dir)
    chunk_files = sorted(chunk_dir.glob("*.txt"))

    if not chunk_files:
        print("No chunk files found.")
        return

    reduced = []

    for chunk_file in chunk_files:
        if not args.reduced_only:
            print(f"===== {chunk_file.name} =====")

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
