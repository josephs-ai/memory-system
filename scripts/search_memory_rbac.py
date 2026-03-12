import argparse
import json
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from memory_access_policy import allowed_sensitivities
from memory_db import hybrid_search_memory_items, close_pool

WORKSPACE = Path.home() / ".openclaw" / "workspace"
MEMORY_DIR = WORKSPACE / "memory"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def collect_project_sources(project_id: str | None):
    if not project_id:
        return

    project_dir = MEMORY_DIR / "projects" / project_id
    current_file = project_dir / "current.md"
    milestone_files = sorted(project_dir.glob("milestone-*.md"))
    daily_dir = project_dir / "daily"
    daily_files = sorted(daily_dir.glob("*.md")) if daily_dir.exists() else []

    if current_file.exists():
        mtime = current_file.stat().st_mtime
        for line in current_file.read_text(encoding="utf-8", errors="replace").splitlines():
            s = line.strip()
            if s.startswith("- "):
                yield {
                    "source_type": "project_current",
                    "path": str(current_file),
                    "text": s,
                    "sensitivity": "internal",
                    "mtime": mtime,
                }

    for mf in milestone_files:
        mtime = mf.stat().st_mtime
        for line in mf.read_text(encoding="utf-8", errors="replace").splitlines():
            s = line.strip()
            if s.startswith("- ") or s.startswith("## ") or s.startswith("### "):
                yield {
                    "source_type": "project_milestone",
                    "path": str(mf),
                    "text": s,
                    "sensitivity": "internal",
                    "mtime": mtime,
                }

    for df in daily_files[-7:]:
        mtime = df.stat().st_mtime
        for line in df.read_text(encoding="utf-8", errors="replace").splitlines():
            s = line.strip()
            if s.startswith("- "):
                yield {
                    "source_type": "project_daily",
                    "path": str(df),
                    "text": s,
                    "sensitivity": "internal",
                    "mtime": mtime,
                }


def simple_project_score(query: str, text: str, source_type: str) -> float:
    q_terms = [x.lower() for x in query.split() if x.strip()]
    t = (text or "").lower()
    raw_hits = sum(1 for term in q_terms if term in t)

    base = raw_hits * 0.30

    # mild source bias: current > milestone > daily
    if source_type == "project_current":
        base += 0.12
    elif source_type == "project_milestone":
        base += 0.07
    elif source_type == "project_daily":
        base += 0.03

    return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--project-id")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    allowed = list(allowed_sensitivities(args.agent))

    model = SentenceTransformer(MODEL_NAME)
    query_embedding = model.encode(
        args.query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    scored = []

    canonical_items = hybrid_search_memory_items(
        query_embedding,
        query_text=args.query,
        status="active",
        allowed_sensitivities=allowed,
        limit=50,
    )

    for item in canonical_items:
        scored.append(
            {
                "score": float(item.get("final_score", 0) or 0) + 0.10,
                "source_type": "canonical",
                "path": "db:memory_items",
                "sensitivity": item.get("sensitivity", "public"),
                "text": item.get("text", ""),
                "mtime": float("inf"),
            }
        )

    for src in collect_project_sources(args.project_id):
        if src["sensitivity"] not in allowed:
            continue
        score = simple_project_score(args.query, src["text"], src["source_type"])
        if score > 0:
            row = dict(src)
            row["score"] = float(score)
            scored.append(row)

    scored.sort(key=lambda x: (-x["score"], -x.get("mtime", 0), x["source_type"], x["path"]))

    if not scored:
        print("NONE")
        close_pool()
        return

    for row in scored[: args.top_k]:
        print(json.dumps({
            "score": row["score"],
            "source_type": row["source_type"],
            "path": row["path"],
            "sensitivity": row["sensitivity"],
            "text": row["text"],
        }, ensure_ascii=False))

    close_pool()


if __name__ == "__main__":
    main()
