import os
import json
import sqlite3
import argparse
import numpy as np
import re
from datetime import datetime
from sentence_transformers import SentenceTransformer

CONFIG_PATH = os.path.expanduser("~/.openclaw/workspace/.memory-index/config.json")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

db_path = config["indexDb"]
model_name = "all-MiniLM-L6-v2"

parser = argparse.ArgumentParser()
parser.add_argument("query", type=str)
parser.add_argument("--topk", type=int, default=5)
args = parser.parse_args()

def tokenize(text):
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())

query_tokens = tokenize(args.query)
query_token_set = set(query_tokens)

model = SentenceTransformer(model_name)
qvec = model.encode([args.query])[0].astype(np.float32)

conn = sqlite3.connect(db_path)
cur = conn.cursor()
cur.execute("""
SELECT
  c.id,
  d.path,
  c.chunk_index,
  c.heading,
  c.text,
  e.vector_path
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
    if denom == 0:
        vector_score = 0.0
    else:
        vector_score = float(np.dot(qvec, vec) / denom)

    text_tokens = tokenize(text)
    text_token_set = set(text_tokens)

    overlap = len(query_token_set & text_token_set)
    keyword_score = overlap / max(1, len(query_token_set))

    final_score = 0.7 * vector_score + 0.3 * keyword_score

    results.append({
        "chunk_id": chunk_id,
        "path": path,
        "chunk_index": chunk_index,
        "heading": heading,
        "text": text,
        "vector_score": vector_score,
        "keyword_score": keyword_score,
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

print(f"Query: {args.query}")
print()

for i, r in enumerate(top, 1):
    print(
        f"[{i}] final={r['final_score']:.4f} "
        f"vector={r['vector_score']:.4f} "
        f"keyword={r['keyword_score']:.4f} "
        f"path={r['path']} chunk={r['chunk_index']}"
    )
    if r["heading"]:
        print(f"    heading: {r['heading']}")
    preview = r["text"][:300].replace("\n", " ")
    print(f"    text: {preview}")
    print()
