"""Stage 2 dev evaluation: retrieval recall, role-leak check, refusal signals.

Runs the published dev questions through the Retriever and reports, tuned on
dev data only (the held-out set is never touched here):

- recall@k of the gold section among role/temporal-eligible hits (answered
  questions), overall and for the temporal and contradiction slices the
  grader scores separately;
- role-isolation invariant: no hit may come from a document above the
  requester's role (asserted on every question, expected 0 violations);
- refusal groundwork: how often role-blocked questions show blocked
  evidence, and the score distribution of out-of-corpus questions, which
  the service stage turns into not_permitted / not_in_corpus decisions.

Usage: python -m policybook.retrieve_eval [--questions data/questions_dev.jsonl]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

from .retrieve import ROLE_RANK, Retriever

REPO_ROOT = Path(__file__).resolve().parent.parent
SLICES = {"temporal": {"temporal_reasoning", "temporal"},
          "contradiction": {"contradiction_detection", "contradiction"}}
KS = (1, 3, 6, 10)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="policybook.retrieve_eval")
    p.add_argument("--questions", default=str(REPO_ROOT / "data" / "questions_dev.jsonl"))
    p.add_argument("--index-dir", default=None)
    p.add_argument("--model-dir", default=str(REPO_ROOT / "models" / "all-MiniLM-L6-v2"))
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--out", default=str(REPO_ROOT / "results" / "retrieval_dev.jsonl"))
    p.add_argument("--report", default=str(REPO_ROOT / "results" / "retrieval_report.json"))
    args = p.parse_args(argv)

    import os
    index_dir = args.index_dir or os.environ.get("INDEX_DIR") or str(REPO_ROOT / ".index")
    questions = [json.loads(l) for l in open(args.questions) if l.strip()]
    retr = Retriever(index_dir, args.model_dir)

    rows, leaks = [], 0
    t0 = time.perf_counter()
    for q in questions:
        r = retr.retrieve(q["question"], q["role"], q["as_of"], k=args.k)
        leaks += sum(1 for h in r.hits
                     if ROLE_RANK[retr.docs[h.doc_id]["role"]] > ROLE_RANK[q["role"]])
        gold_sections = {c["section"] for c in (q.get("gold_citations") or [])}
        got_sections = [h.section for h in r.hits]
        rank = next((i + 1 for i, s in enumerate(got_sections) if s in gold_sections), None)
        rows.append({
            "id": q["id"], "expected_status": q["expected_status"], "role": q["role"],
            "as_of": q["as_of"], "tags": q.get("tags", []),
            "gold_rank": rank, "blocked_evidence": r.blocked_evidence,
            "allowed_pool": r.allowed_pool,
            "top_score": r.hits[0].score if r.hits else 0.0,
            "top_cosine": r.top_cosine, "top_bm25": r.top_bm25,
            "hits": [{"section": h.section, "doc_id": h.doc_id, "score": h.score,
                      "vec_rank": h.vec_rank, "bm25_rank": h.bm25_rank}
                     for h in r.hits]})
    elapsed = time.perf_counter() - t0

    def recall(sub, k):
        ans = [x for x in sub if x["expected_status"] == "answered"]
        return round(sum(1 for x in ans if x["gold_rank"] and x["gold_rank"] <= k)
                     / len(ans), 4) if ans else None

    summary = {
        "questions": len(rows), "k_max": args.k, "seconds": round(elapsed, 2),
        "role_leak_violations": leaks,
        "recall": {f"@{k}": recall(rows, k) for k in KS},
        "slices": {name: {f"recall@{k}": recall([x for x in rows if set(x["tags"]) & tags], k)
                          for k in (6,)} | {"n": len([x for x in rows if set(x["tags"]) & tags and x["expected_status"] == "answered"])}
                   for name, tags in SLICES.items()},
        "refusal_signals": {
            "not_permitted_with_blocked_evidence": (
                f"{sum(1 for x in rows if x['expected_status'] == 'not_permitted' and x['blocked_evidence'])}"
                f"/{sum(1 for x in rows if x['expected_status'] == 'not_permitted')}"),
            "not_in_corpus_top_scores": sorted(
                (x["top_score"] for x in rows if x["expected_status"] == "not_in_corpus"),
                reverse=True),
        },
    }
    if leaks:
        print(f"ROLE LEAK: {leaks} above-role hits returned", file=sys.stderr)
        return 1
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    Path(args.report).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
