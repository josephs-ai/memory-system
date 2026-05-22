"""
Memory system utility: retrieve memory hybrid boosted.

Key functions: tok, boost
"""
import os
import json
import sqlite3
import argparse
import numpy as np
import re
from datetime import datetime
from sentence_transformers import SentenceTransformer

CONFIG_PATH = os.environ.get("OPENCLAW_MEMORY_CONFIG", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json"))
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

db_path = config["indexDb"]
model_name = "all-MiniLM-L6-v2"

parser = argparse.ArgumentParser()
parser.add_argument("query", type=str)
parser.add_argument("--topk", type=int, default=5)
args = parser.parse_args()

def tok(text):
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())

def boost(query_tokens, path):
    q = set(query_tokens)
    p = path.lower()
    b = 0.0

    browser_terms = {"browser","cdp","chromium","devtools","9222","9223","9224","visible","headless","port"}
    gateway_terms = {"gateway","nginx","bind","token","18789","loopback","worker","upload","couchdb","vnc"}
    pref_terms = {"preference","workflow","prefer"}
    fix_terms = {"fix","error","issue","broken","failed","auth"}

    if q & browser_terms and "memory/browser.md" in p:
        b += 0.35
    if q & gateway_terms and "memory/architecture.md" in p:
        b += 0.35
    if q & pref_terms and "memory/preferences.md" in p:
        b += 0.20
    if q & fix_terms and "memory/learned-fixes.md" in p:
        b += 0.20

    if q & gateway_terms and "memory/projects/" in p:
        b -= 0.20
    if q & browser_terms and "memory/projects/" in p:
        b -= 0.10
    if ("memory/daily/" in p) or ("/memory/projects/" in p and "/daily/" in p):
        b -= 0.15

    return b

query_tokens = tok(args.query)
query_token_set = set(query_tokens)

model = SentenceTransformer(model_name)
qvec = model.encode([args.query])[0].astype(np.float32)

conn = sqlite3.connect(db_path)
cur = conn.cursor()

cur.execute("""
SELECT c.id, d.path, c.chunk_index, c.heading, c.text, e.vector_path
FROM chunks c
JOIN documents d ON d.id = c.document_id
JOIN embeddings e ON e.chunk_id = c.id
ORDER BY c.id
""")
rows = cur.fetchall()
results = []
for chunk_id, path, chunk_index, heading, text, vector_path in rows:
    vec = np.load(vector_path).astype(np.float32)
    denom = (np.linalg.norm(qvec) * np.linalg.norm(vec))
    vector_score = 0.0 if denom == 0 else float(np.dot(qvec, vec) / denom)

    text_token_set = set(tok(text))
    overlap = len(query_token_set & text_token_set)
    keyword_score = overlap / max(1, len(query_token_set))

    path_boost = boost(query_tokens, path)
    final_score = 0.5 * vector_score + 0.3 * keyword_score + 0.2 * path_boost

    results.append({
        "path": path,
        "chunk_index": chunk_index,
        "heading": heading,
        "text": text,
        "vector_score": vector_score,
        "keyword_score": keyword_score,
        "boost": path_boost,
        "final_score": final_score
    })

results.sort(key=lambda x: x["final_score"], reverse=True)
top = results[:args.topk]

now = datetime.utcnow().isoformat() + "Z"
cur.execute(
    "INSERT INTO retrieval_log(query, result_count, created_at) VALUES (?, ?, ?)",
    (args.query, len(top), now)
)
conn.commit()
conn.close()

print(f"Query: {args.query}\n")
for i, r in enumerate(top, 1):
    print(f"[{i}] final={r['final_score']:.4f} vector={r['vector_score']:.4f} keyword={r['keyword_score']:.4f} boost={r['boost']:.4f} path={r['path']} chunk={r['chunk_index']}")
    if r["heading"]:
        print(f"    heading: {r['heading']}")
    print(f"    text: {r['text'][:220].replace(chr(10), ' ')}\n")
