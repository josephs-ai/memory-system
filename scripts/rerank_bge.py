import argparse
from FlagEmbedding import FlagReranker

MODEL_NAME = "BAAI/bge-reranker-base"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--text", action="append", required=True)
    args = parser.parse_args()

    reranker = FlagReranker(MODEL_NAME, use_fp16=False)
    pairs = [[args.query, t] for t in args.text]
    scores = reranker.compute_score(pairs)

    rows = list(zip(args.text, scores))
    rows.sort(key=lambda x: x[1], reverse=True)

    for text, score in rows:
        print({"score": float(score), "text": text})

if __name__ == "__main__":
    main()
