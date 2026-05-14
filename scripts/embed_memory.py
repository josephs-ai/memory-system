"""Generate and store vector embeddings for memory."""
import os
import json
import sqlite3
import numpy as np
from datetime import datetime
from pathlib import Path
from sentence_transformers import SentenceTransformer

CONFIG_PATH = os.environ.get("OPENCLAW_MEMORY_CONFIG", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json"))

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

db_path = config["indexDb"]
embeddings_dir = Path(config["embeddingsDir"])
embeddings_dir.mkdir(parents=True, exist_ok=True)

model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

conn = sqlite3.connect(db_path)
cur = conn.cursor()

cur.execute("""
SELECT c.id, c.text
FROM chunks c
LEFT JOIN embeddings e ON e.chunk_id = c.id
WHERE e.chunk_id IS NULL
ORDER BY c.id
""")
rows = cur.fetchall()

now = datetime.utcnow().isoformat() + "Z"

count = 0
for chunk_id, text in rows:
    vec = model.encode([text])[0]
    vec = np.asarray(vec, dtype=np.float32)

    vec_path = embeddings_dir / f"{chunk_id}.npy"
    np.save(vec_path, vec)

    cur.execute("""
        INSERT OR REPLACE INTO embeddings(chunk_id, model, vector_path, created_at)
        VALUES (?, ?, ?, ?)
    """, (chunk_id, model_name, str(vec_path), now))

    count += 1

conn.commit()
conn.close()

print(f"Embedded {count} chunks using {model_name}.")
