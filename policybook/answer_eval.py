"""Stage 3 dev evaluation: run the /ask pipeline over the dev questions and
grade the answers with the vendored published grader.

Default is deterministic checks only (grader --no-judge): schema validity,
coverage, refusal rates, verbatim citation rate, role violations. The
LLM-judged metrics (answer correctness, citation support) run only when a
budget-class judge is configured (LLM_BASE_URL/LLM_API_KEY/JUDGE_MODEL) -
never assumed, never claimed without a measured run.

Dev split only; the held-out set is never read here.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .answer import Answerer
from .llm import QuotaExceeded

REPO_ROOT = Path(__file__).resolve().parent.parent
GRADER = REPO_ROOT / "tools" / "op05_grader" / "grader.py"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="policybook.answer_eval")
    p.add_argument("--questions", default=str(REPO_ROOT / "data" / "questions_dev.jsonl"))
    p.add_argument("--out", default=str(REPO_ROOT / "results" / "answers_dev.jsonl"))
    p.add_argument("--report", default=str(REPO_ROOT / "results" / "grader_dev_report.json"))
    p.add_argument("--judge", action="store_true",
                   help="also run the rubric LLM judge (needs LLM_BASE_URL/LLM_API_KEY/JUDGE_MODEL)")
    args = p.parse_args(argv)

    corpus_dir = os.environ.get("CORPUS_DIR", str(REPO_ROOT / "data" / "corpus"))
    index_dir = os.environ.get("INDEX_DIR", str(REPO_ROOT / ".index"))
    model_dir = os.environ.get("MODEL_DIR", str(REPO_ROOT / "models" / "all-MiniLM-L6-v2"))
    questions = [json.loads(l) for l in open(args.questions) if l.strip()]
    ans = Answerer(index_dir, model_dir, corpus_dir)

    rows, lat = [], []
    tok_in = tok_out = gen_calls = fallbacks = 0
    cost_total = 0.0
    t0 = time.perf_counter()
    try:
        for q in questions:
            # Quota policy: stop on the first rate/quota signal and report
            # partial evidence - no automatic retries, no silent fallback
            # conversions (a partial run is labeled, never claimed whole).
            res = ans.answer(q["id"], q["question"], q["role"], q.get("as_of", "9999-12-31"))
            if ans.gen_backend != "extractive":
                time.sleep(float(os.environ.get("OP05_GEN_DELAY_SEC", "1.0")))
            lat.append(res.latency_ms)
            rows.append(res.public_dict())
            cost_total += res.cost_usd
            if res.generator.startswith("llm:"):
                u = res.debug.get("gen_usage", {})
                tok_in += u.get("input_tokens", 0)
                tok_out += u.get("output_tokens", 0)
                gen_calls += u.get("calls", 0)
            elif ans.gen_backend != "extractive" and res.status == "answered":
                fallbacks += 1
    except QuotaExceeded as e:
        partial = Path(args.out).with_suffix(".partial.jsonl")
        with open(partial, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"ABORTED on provider rate/quota after {len(rows)} questions: {e}",
              file=sys.stderr)
        print(f"Partial rows ({len(rows)}) written to {partial} - labeled "
              "partial evidence, not a completed run. Stop; do not retry "
              "automatically.", file=sys.stderr)
        return 3
    elapsed = time.perf_counter() - t0
    lat.sort()
    p95 = lat[int(0.95 * (len(lat) - 1))]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    by_status = {}
    for q, r in zip(questions, rows):
        key = (q["expected_status"], r["status"])
        by_status[key] = by_status.get(key, 0) + 1
    run_summary = {
        "questions": len(rows), "seconds": round(elapsed, 2),
        "latency_ms": {"p50": round(lat[len(lat) // 2], 1), "p95": round(p95, 1),
                       "max": round(lat[-1], 1)},
        "status_confusion_expected_vs_got": {f"{a}->{b}": n for (a, b), n in sorted(by_status.items())},
        "generator": (f"llm:{ans.gen_model} via {ans.gen_backend}"
                      if ans.gen_backend != "extractive"
                      else "extractive (deterministic, 0 API tokens)"),
        "gen_api_calls": gen_calls,
        "tokens": {"input": tok_in, "output": tok_out,
                   "avg_input_per_llm_answer": round(tok_in / max(1, sum(
                       1 for r in rows if r["status"] == "answered")), 1),
                   "avg_output_per_llm_answer": round(tok_out / max(1, sum(
                       1 for r in rows if r["status"] == "answered")), 1)},
        "cost_usd_at_list_price": round(cost_total, 6),
        "reference_tokens_per_question": {"input": 1005, "output": 743},
        "extractive_fallbacks": fallbacks,
    }

    cmd = [sys.executable, str(GRADER), "--pred", args.out, "--gold", args.questions,
           "--corpus", str(Path(corpus_dir) / "manifest.jsonl"),
           "--report", args.report]
    if not args.judge:
        cmd.append("--no-judge")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        return proc.returncode
    run_summary["grader_stdout"] = json.loads(proc.stdout)
    Path(args.report).with_suffix(".run.json").write_text(json.dumps(run_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
