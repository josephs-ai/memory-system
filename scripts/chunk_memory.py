"""
Memory system utility: chunk memory.

Key functions: split_words, chunk_text, detect_heading
"""
import os
import re
import json
import glob
import sqlite3
from datetime import datetime
from pathlib import Path

CONFIG_PATH = os.environ.get("OPENCLAW_MEMORY_CONFIG", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json"))

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

root = Path(config["sourceMemoryRoot"])
db_path = config["indexDb"]
chunks_dir = Path(config["chunksDir"])
chunk_target_words = config["chunkTargetWords"]
chunk_overlap_words = config["chunkOverlapWords"]

source_paths = []
for rel in config["sourceFiles"]:
    p = root / rel
    if p.exists():
        source_paths.append(p)

for pattern in config["sourceGlobs"]:
    for match in glob.glob(str(root / pattern), recursive=True):
        p = Path(match)
        if p.exists() and p.is_file():
            source_paths.append(p)

# dedupe
source_paths = sorted(set(source_paths))

def split_words(text):
    return re.findall(r"\S+", text)

def chunk_text(text, target_words=180, overlap_words=40):
    words = split_words(text)
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + target_words, len(words))
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
        start = max(end - overlap_words, start + 1)
    return chunks

def detect_heading(text):
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return ""

conn = sqlite3.connect(db_path)
cur = conn.cursor()

now = datetime.utcnow().isoformat() + "Z"

for path in source_paths:
    rel_path = str(path.relative_to(root))
    text = path.read_text(encoding="utf-8", errors="ignore").strip()

    kind = "memory"
    if rel_path.startswith("memory/daily/"):
        kind = "daily"
    elif rel_path.startswith("memory/projects/") and "/daily/" in rel_path:
        kind = "daily"
    elif rel_path.startswith("memory/projects/"):
        kind = "project"

    elif rel_path == "MEMORY.md":
        kind = "core"
    elif rel_path.startswith("memory/"):
        kind = "stable"

    cur.execute("INSERT OR IGNORE INTO documents(path, kind, updated_at) VALUES (?, ?, ?)",
                (rel_path, kind, now))
    cur.execute("UPDATE documents SET kind = ?, updated_at = ? WHERE path = ?",
                (kind, now, rel_path))
    cur.execute("SELECT id FROM documents WHERE path = ?", (rel_path,))
    document_id = cur.fetchone()[0]

    cur.execute("DELETE FROM embeddings WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id = ?)", (document_id,))
    cur.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))

    heading = detect_heading(text)
    chunks = chunk_text(text, chunk_target_words, chunk_overlap_words)

    for idx, chunk in enumerate(chunks):
        token_estimate = max(1, len(split_words(chunk)))
        cur.execute("""
            INSERT INTO chunks(document_id, chunk_index, heading, text, token_estimate, importance, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (document_id, idx, heading, chunk, token_estimate, 0.5, now, now))

        chunk_id = cur.lastrowid
        chunk_file = chunks_dir / f"{chunk_id}.txt"
        chunk_file.write_text(chunk, encoding="utf-8")

conn.commit()
conn.close()

print(f"Indexed {len(source_paths)} documents into chunks.")
