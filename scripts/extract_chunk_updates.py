import json
import argparse
import subprocess
import tempfile
from pathlib import Path

WORKSPACE = Path.home() / ".openclaw" / "workspace"
SCRIPTS_DIR = WORKSPACE / ".memory-index" / "scripts"

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


def write_temp_text(text: str, suffix: str) -> Path:
    tmp = Path(tempfile.mkstemp(suffix=suffix)[1])
    tmp.write_text(text + ("\n" if text and not text.endswith("\n") else ""), encoding="utf-8")
    return tmp


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
            item.get("memory_type"),
            item.get("entity"),
            item.get("property"),
            item.get("value"),
            item.get("text"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-dir", required=True)
    parser.add_argument("--source-agent", default="unknown")
    parser.add_argument("--source-session", default="unknown")
    parser.add_argument("--reduced-only", action="store_true")
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

        generated = run_stage([
            "python3",
            str(GENERATE_SCRIPT),
            str(chunk_file),
            "--source-agent", args.source_agent,
            "--source-session", args.source_session,
            "--source-chunk", chunk_file.name,
        ])

        if not generated or generated == "NONE":
            if not args.reduced_only:
                print("NONE")
                print()
            continue

        generated_tmp = write_temp_text(generated, ".generated.jsonl")

        filtered = run_stage([
            "python3",
            str(FILTER_SCRIPT),
            str(generated_tmp),
        ])

        if not filtered or filtered == "NONE":
            if not args.reduced_only:
                print("NONE")
                print()
            try:
                generated_tmp.unlink()
            except Exception:
                pass
            continue

        filtered_tmp = write_temp_text(filtered, ".filtered.jsonl")

        judged = run_stage([
            "python3",
            str(JUDGE_SCRIPT),
            str(filtered_tmp),
        ])


        if not judged or judged == "NONE":
            if not args.reduced_only:
                print("NONE")
                print()
            try:
                generated_tmp.unlink()
            except Exception:
                pass
            try:
                filtered_tmp.unlink()
            except Exception:
                pass
            continue

        items = load_jsonl_text(judged)
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

            for item in items:
                reduced.append(item)
        try:
            generated_tmp.unlink()
        except Exception:
            pass
        try:
            filtered_tmp.unlink()
        except Exception:
            pass

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
