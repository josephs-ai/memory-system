"""
locomo/run_reme.py — Drive ReMe (reme-ai) over the SAME LoCoMo slice as ours.

Fair-comparison contract (matches the bake-off plan):
  * SAME 2 conversations, SAME first-N QA per conv (same question-id list).
  * SAME answer LLM: ReMe's generation is pinned to gpt-4o via env (LLM_MODEL_NAME),
    and the answer is produced by OUR reader (reader.read_answer_str) over ReMe's
    retrieved chunks — so the only thing that differs from "ours" is the
    MEMORY + RETRIEVAL layer (ReMe vs our typed-lane store). The generator + prompt
    + judge are byte-for-byte identical.
  * SAME judge: common.llm_judge_answer(strict_judge=True, model=openai/gpt-4o),
    same >=0.5 binarization, same abstention rubric (cat-5 gold == NO_ANSWER_SENTINEL).

ReMe must already be running (reme start ...) on REME_HOST:REME_PORT with the
pinned LLM/embedding env. This script is a CLIENT: it shells out to `reme` for
auto_memory (ingest) and search (retrieve).

Usage:
    python locomo/run_reme.py --max-convs 2 --max-qa 10 --save
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCHMARKS_DIR))
sys.path.insert(0, str(BENCHMARKS_DIR.parent / "scripts"))

from reader import read_answer_str
from common import (
    JudgeError,
    NO_ANSWER_SENTINEL,
    compute_percentiles,
    count_tokens_list,
    exact_match,
    llm_judge_answer,
    save_results,
    token_f1,
)
from locomo.adapter import get_qa_pairs, load_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger("openclaw.benchmarks.locomo.reme")

REME_BIN = os.environ.get("REME_BIN", "/tmp/reme-venv/bin/reme")
JUDGE_CORRECT_THRESHOLD = 0.5
JUDGE_ERROR_INVALID_RATE = 0.02
ABSTAIN_PREDICTION = "I cannot determine the answer from the available information."


def _reme(action: str, timeout: int = 180, **kwargs) -> str:
    """Call the reme client and return raw stdout."""
    cmd = [REME_BIN, action]
    for k, v in kwargs.items():
        if isinstance(v, (dict, list)):
            v = json.dumps(v)
        cmd.append(f"{k}={v}")
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0:
        raise RuntimeError(f"reme {action} failed (rc={res.returncode}): {res.stderr[-500:]}")
    return res.stdout


def _parse_reme_json(stdout: str) -> dict:
    """ReMe prints a human line then a final '✅ {json}' line. Extract the JSON."""
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("✅"):
            line = line[1:].strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    # Fallback: find the last {...}
    start = stdout.rfind("{")
    if start >= 0:
        try:
            return json.loads(stdout[start:])
        except json.JSONDecodeError:
            pass
    return {}


def _flatten_turns(conversation: dict | list) -> list[dict]:
    """Reuse the adapter's flattening logic to get ordered (speaker, text, date) turns."""
    speaker_a = conversation.get("speaker_a", "A") if isinstance(conversation, dict) else "A"
    speaker_b = conversation.get("speaker_b", "B") if isinstance(conversation, dict) else "B"
    turns = []
    if isinstance(conversation, dict):
        idx = 1
        while True:
            sess = conversation.get(f"session_{idx}")
            if sess is None:
                break
            date = conversation.get(f"session_{idx}_date_time", "")
            if isinstance(sess, list):
                for t in sess:
                    t = dict(t)
                    t["_session"] = idx
                    t["_date"] = date
                    turns.append(t)
            idx += 1
    elif isinstance(conversation, list):
        turns = conversation

    out = []
    for t in turns:
        text = (t.get("text") or "").strip()
        if not text:
            continue
        sid = t.get("speaker", t.get("id", "unknown"))
        if sid in ("A", "a"):
            spk = speaker_a
        elif sid in ("B", "b"):
            spk = speaker_b
        else:
            spk = sid
        out.append({"speaker": spk, "text": text, "date": t.get("_date", t.get("date", ""))})
    return out


def _ingest_conversation(conv_id: str, turns: list[dict], session_prefix: str) -> int:
    """Ingest a conversation into ReMe via auto_memory, batching turns into messages.

    ReMe's auto_memory records conversation facts into a daily note. We push the
    turns in batches as a multi-message conversation so ReMe extracts memory facts.
    """
    n = 0
    BATCH = 12
    for i in range(0, len(turns), BATCH):
        batch = turns[i:i + BATCH]
        messages = []
        for t in batch:
            content = t["text"]
            if t["date"]:
                content = f"[{t['date']}] {content}"
            messages.append({"role": "user", "name": t["speaker"], "content": content})
        sess = f"{session_prefix}_{i // BATCH}"
        try:
            _reme("auto_memory", timeout=240, messages=messages, session_id=sess)
            n += len(batch)
        except Exception as e:
            LOGGER.warning("auto_memory batch %d failed for %s: %s", i // BATCH, conv_id, e)
    return n


def _reme_search_to_items(query: str, limit: int = 10) -> tuple[list[dict], float]:
    """Run ReMe search; map its chunks into our reader's item dict shape."""
    t0 = time.time()
    out = _reme("search", timeout=120, query=query, limit=limit)
    latency = time.time() - t0
    data = _parse_reme_json(out)
    items = []
    for r in data.get("results", []):
        text = r.get("text", "")
        items.append({
            "text": text,
            "value": text[:200],
            "entity": "",
            "property": "",
            "last_confirmed": "",
            "first_seen": "",
            "score": (r.get("scores") or {}).get("score", 0.0),
        })
    return items, latency


def run_reme_locomo(max_convs: int, max_qa: int, do_judge: bool, judge_model: str,
                    skip_ingest: bool = False) -> dict:
    data = load_dataset(max_convs=max_convs)
    LOGGER.info("ReMe driver: %d conversations (judge=%s, skip_ingest=%s)",
                len(data), do_judge, skip_ingest)

    all_results = []
    latencies: list[float] = []
    token_counts: list[int] = []
    f1_scores: list[float] = []
    em_scores: list[float] = []
    category_stats: dict[str, dict] = {}

    for conv_idx, (conv_id, conv_data) in enumerate(data.items()):
        conversation = conv_data.get("conversation", conv_data.get("dialog", []))
        qa_pairs = get_qa_pairs(conv_data)
        if not conversation or not qa_pairs:
            continue
        qa_pairs = qa_pairs[:max_qa]

        turns = _flatten_turns(conversation)
        if skip_ingest:
            LOGGER.info("[%d/%d] conv=%s turns=%d qa=%d -> SKIP ingest (using existing vault)",
                        conv_idx + 1, len(data), conv_id, len(turns), len(qa_pairs))
        else:
            run_id = f"reme_{conv_id}_{uuid.uuid4().hex[:6]}"
            LOGGER.info("[%d/%d] conv=%s turns=%d qa=%d -> ingesting into ReMe",
                        conv_idx + 1, len(data), conv_id, len(turns), len(qa_pairs))
            ingested = _ingest_conversation(conv_id, turns, run_id)
            LOGGER.info("  ingested %d turns; waiting for ReMe index", ingested)
            time.sleep(12)  # let the watch/index loop pick up the new notes

        for qa in qa_pairs:
            question = qa["question"]
            gold_answer = str(qa["answer"])
            category = qa["category"]
            is_unanswerable = bool(qa.get("unanswerable")) or gold_answer == NO_ANSWER_SENTINEL
            qid = f"{conv_id}:{len(all_results)}"

            items, latency = _reme_search_to_items(question, limit=10)
            latencies.append(latency)
            token_counts.append(count_tokens_list([it["text"] for it in items]))

            llm_answer = read_answer_str(question, items)
            abstained = llm_answer is None
            if llm_answer is not None:
                predicted = llm_answer
            elif is_unanswerable:
                predicted = ABSTAIN_PREDICTION
            else:
                predicted = ""  # no answer extracted

            f1 = token_f1(predicted, gold_answer)
            em = exact_match(predicted, gold_answer)
            f1_scores.append(f1)
            em_scores.append(em)

            judge_score = None
            judged_correct = None
            judge_error = False
            judge_error_msg = None
            if do_judge:
                try:
                    judge_score = llm_judge_answer(
                        question, gold_answer, predicted,
                        model=judge_model, strict_judge=True,
                    )
                    judged_correct = judge_score >= JUDGE_CORRECT_THRESHOLD
                except JudgeError as e:
                    judge_error = True
                    judge_error_msg = str(e)
                    LOGGER.warning("judge_error on %s: %s", qid, judge_error_msg)

            cs = category_stats.setdefault(
                category,
                {"f1_sum": 0.0, "em_sum": 0.0, "count": 0,
                 "judge_score_sum": 0.0, "judge_correct": 0, "judge_scored": 0,
                 "judge_error": 0, "unanswerable": is_unanswerable},
            )
            cs["f1_sum"] += f1
            cs["em_sum"] += em
            cs["count"] += 1
            if do_judge:
                if judge_error:
                    cs["judge_error"] += 1
                else:
                    cs["judge_score_sum"] += judge_score
                    cs["judge_scored"] += 1
                    if judged_correct:
                        cs["judge_correct"] += 1

            all_results.append({
                "conv_id": conv_id, "question": question, "gold_answer": gold_answer,
                "predicted": predicted, "category": category,
                "unanswerable": is_unanswerable, "abstained": abstained,
                "f1": f1, "exact_match": em,
                "judge_score": judge_score, "judged_correct": judged_correct,
                "judge_error": judge_error, "judge_error_msg": judge_error_msg,
                "latency": latency, "retrieved_count": len(items),
            })

    if not all_results:
        return {"benchmark": "LoCoMo-ReMe", "score": 0.0, "error": "no results"}

    lat_stats = compute_percentiles(latencies)
    out = {
        "benchmark": "LoCoMo-ReMe",
        "system": "ReMe (under our harness)",
        "avg_f1": sum(f1_scores) / len(f1_scores),
        "avg_exact_match": sum(em_scores) / len(em_scores),
        "tokens_avg": sum(token_counts) / max(1, len(token_counts)),
        "latency_p50": lat_stats["p50"],
        "latency_p95": lat_stats["p95"],
        "num_qa_pairs": len(all_results),
        "num_conversations": len(data),
        "per_question": all_results,
    }

    cat_summary = {}
    for cat, v in category_stats.items():
        entry = {"avg_f1": v["f1_sum"] / v["count"], "avg_em": v["em_sum"] / v["count"],
                 "count": v["count"], "unanswerable": v.get("unanswerable", False)}
        if do_judge:
            scored = v["judge_scored"]
            entry["judge_scored"] = scored
            entry["judge_error"] = v["judge_error"]
            entry["judge_accuracy"] = (v["judge_correct"] / scored) if scored else None
            entry["judge_mean_float"] = (v["judge_score_sum"] / scored) if scored else None
        cat_summary[cat] = entry
    out["by_category"] = cat_summary

    if do_judge:
        ans_c = ans_s = abs_c = abs_s = 0
        ans_f = abs_f = 0.0
        errs = 0
        for r in all_results:
            if r["judge_error"]:
                errs += 1
                continue
            if r["judge_score"] is None:
                continue
            if r["unanswerable"]:
                abs_s += 1; abs_f += r["judge_score"]
                if r["judged_correct"]: abs_c += 1
            else:
                ans_s += 1; ans_f += r["judge_score"]
                if r["judged_correct"]: ans_c += 1
        total = len(all_results)
        rate = errs / max(1, total)
        out["judge"] = {
            "judge_model": judge_model, "strict_judge": True,
            "correct_threshold": JUDGE_CORRECT_THRESHOLD,
            "total_items": total, "judge_errors": errs, "judge_error_rate": rate,
            "invalid": rate > JUDGE_ERROR_INVALID_RATE,
            "answerable": {"scored": ans_s, "accuracy": (ans_c / ans_s) if ans_s else None,
                           "mean_float": (ans_f / ans_s) if ans_s else None},
            "abstention": {"scored": abs_s, "accuracy": (abs_c / abs_s) if abs_s else None,
                           "mean_float": (abs_f / abs_s) if abs_s else None},
            "overall": {"scored": ans_s + abs_s,
                        "accuracy": ((ans_c + abs_c) / (ans_s + abs_s)) if (ans_s + abs_s) else None,
                        "mean_float": ((ans_f + abs_f) / (ans_s + abs_s)) if (ans_s + abs_s) else None},
        }
        out["score"] = out["judge"]["answerable"]["accuracy"] if ans_s else out["judge"]["overall"]["accuracy"]

    return out


def main():
    p = argparse.ArgumentParser(description="ReMe LoCoMo driver (under our harness)")
    p.add_argument("--max-convs", type=int, default=2)
    p.add_argument("--max-qa", type=int, default=10)
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--judge-model", type=str, default="openai/gpt-4o")
    p.add_argument("--skip-ingest", action="store_true",
                   help="Use the already-populated ReMe vault; do not re-ingest conversations.")
    p.add_argument("--save", action="store_true")
    args = p.parse_args()

    # Confirm ReMe is serving.
    try:
        v = _parse_reme_json(_reme("version", timeout=30))
        LOGGER.info("ReMe version: %s", v.get("version"))
    except Exception as e:
        LOGGER.error("ReMe not reachable: %s", e)
        sys.exit(2)

    results = run_reme_locomo(args.max_convs, args.max_qa, not args.no_judge,
                              args.judge_model, skip_ingest=args.skip_ingest)

    print("\n--- LoCoMo (ReMe under our harness) ---")
    print(f"  QA Pairs: {results.get('num_qa_pairs', 0)}")
    print(f"  Avg F1 (sec): {results.get('avg_f1', 0):.3f}  EM: {results.get('avg_exact_match', 0):.3f}")
    print(f"  Latency p50: {results.get('latency_p50', 0)*1000:.1f}ms  p95: {results.get('latency_p95', 0)*1000:.1f}ms")
    j = results.get("judge")
    if j:
        def _f(x): return f"{x:.3f}" if x is not None else "n/a"
        print(f"  Answerable: acc={_f(j['answerable']['accuracy'])} mean_float={_f(j['answerable']['mean_float'])} n={j['answerable']['scored']}")
        print(f"  Abstention: acc={_f(j['abstention']['accuracy'])} mean_float={_f(j['abstention']['mean_float'])} n={j['abstention']['scored']}")
        print(f"  Overall:    acc={_f(j['overall']['accuracy'])} mean_float={_f(j['overall']['mean_float'])} n={j['overall']['scored']}")
        print(f"  judge_errors: {j['judge_errors']}/{j['total_items']} (rate={j['judge_error_rate']*100:.2f}%) => {'INVALID' if j['invalid'] else 'valid'}")
    print("\n  By category:")
    for cat, v in results.get("by_category", {}).items():
        ja = v.get("judge_accuracy"); jm = v.get("judge_mean_float")
        ja_s = f"{ja:.3f}" if ja is not None else "n/a"
        jm_s = f"{jm:.3f}" if jm is not None else "n/a"
        print(f"    cat={cat:<3} judge_acc={ja_s} mean_float={jm_s} f1={v['avg_f1']:.3f} n={v['count']}")

    if args.save:
        path = save_results("locomo_reme", results)
        print(f"\nSaved to: {path}")
    return results


if __name__ == "__main__":
    main()
