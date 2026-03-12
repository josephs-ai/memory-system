import argparse
from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--text", action="append", required=True)
    args = parser.parse_args()

    model = CrossEncoder(MODEL_NAME)
    pairs = [[args.query, t] for t in args.text]
    scores = model.predict(pairs)

    rows = list(zip(args.text, scores))
    rows.sort(key=lambda x: float(x[1]), reverse=True)

    for text, score in rows:
        print({"score": float(score), "text": text})


if __name__ == "__main__":
    main()
