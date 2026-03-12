import argparse
import re
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def normalize_text(s: str) -> str:
    return " ".join(s.strip().split())

def read_bullets(path: Path):
    if not path.exists():
        return []
    bullets = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if s.startswith("- "):
            bullets.append(s[2:].strip())
    return bullets

def cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def specificity_score(text: str) -> float:
    score = 0.0
    t = text.lower()

    if re.search(r"\b\d+\b", t):
        score += 0.2
    if any(x in t for x in ["port", "cdp", "devtools", "token", "loopback", "browser", "gateway", "worker"]):
        score += 0.2
    if any(x in t for x in ["restricted", "indirectly", "through cli", "supports", "managed through", "latest", "current"]):
        score += 0.2

    score += min(len(text) / 300.0, 0.4)
    return score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--text", required=True)
    args = parser.parse_args()

    target = Path(args.target)
    new_text = normalize_text(args.text)
    existing = read_bullets(target)

    if not existing:
        print("NEW")
        print("reason: target file has no existing bullet facts")
        return

    # exact normalized duplicate check first
    for old in existing:
        if normalize_text(old) == new_text:
            print("DUPLICATE")
            print(f"match: {old}")
            print("score: 1.0000")
            return

    new_vec = model.encode([new_text])[0].astype(np.float32)
    old_vecs = model.encode(existing).astype(np.float32)

    scored = []
    for old, vec in zip(existing, old_vecs):
        sim = cosine(new_vec, vec)
        scored.append((sim, old))

    scored.sort(reverse=True, key=lambda x: x[0])
    best_sim, best_old = scored[0]

    new_spec = specificity_score(new_text)
    old_spec = specificity_score(best_old)

    print(f"best_match: {best_old}")
    print(f"similarity: {best_sim:.4f}")
    print(f"new_specificity: {new_spec:.4f}")
    print(f"old_specificity: {old_spec:.4f}")

    if best_sim >= 0.92:
        print("DUPLICATE")
    elif best_sim >= 0.75:
        if new_spec > old_spec + 0.15:
            print("EVOLVE_REPLACE")
        elif old_spec > new_spec + 0.15:
            print("DUPLICATE")
        else:
            print("EVOLVE_APPEND")
    else:
        print("NEW")

if __name__ == "__main__":
    main()
