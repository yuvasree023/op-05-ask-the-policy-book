# EXPERIMENT LOG

## 2026-10-18 — Stage 1 ingestion pipeline (dev environment: 2 vCPU, 2 GB RAM sandbox, Python 3.10)

Built the ingestion stage: corpus validation, section-marker chunking,
vendored quantized MiniLM embeddings, Chroma load with role/temporal
metadata, corpus-hash keyed rebuild.

Run against the shipped Season 1 corpus (`references/OP-05/corpus`,
commit-pinned bytes, all 34 doc sha256 verified against the manifest):

| metric | value |
|---|---|
| documents | 34 |
| sections parsed | 391 (matches published count exactly) |
| passages.jsonl cross-check | 0 id mismatches, 0 text mismatches |
| chunks after splitting | 396 (5 sections over 1000 chars split) |
| embedding dim | 384 |
| embed time | 13.2s |
| index time | 0.3s |
| unchanged-corpus rerun | skipped, 0.14s |
| metadata-only amendment (effective_date flip) | rebuild triggered, new hash |

Unit tests: 4 passed (synthetic corpus: parse/roles, sha256 tamper rejection,
missing-manifest rejection, full ingest cycle + idempotency + swap rebuild).

Nothing tuned yet; no retrieval/generation stage exists. Dev-question
threshold tuning is explicitly deferred to its own stage per the plan.

## 2026-10-18 — Delivery verification (clean extraction, exact shipped archive)

The delivery tarball was extracted to an empty directory and run with only
the documented command (`./scripts/reproduce.sh`), not from the build
workspace:

| check | result |
|---|---|
| first run, fresh venv (deps from PyPI) | OK in 54.7s total; ingestion 14.1s; 34 docs / 391 sections / 396 chunks; validation [] |
| rerun, unchanged corpus | skipped, 0.12s |
| CORPUS_DIR override with one amended doc (effective_date) | rebuild triggered, new corpus hash |
| forced full run with PyPI unreachable (dead index) + HF offline env | OK, 13.6s - model and corpus need no network |
| fresh venv install from vendored wheelhouse with PyPI unreachable | OK, full ingestion 13.3s, 34/391/396, validation [] |

The wheelhouse test used `scripts/vendor_wheels.sh` output (91 wheels,
~102 MB) placed at `vendor/wheels/`; reproduce.sh selected `--no-index`
install automatically.

## 2026-10-18 — Reproduction footprint audit (judge-environment concerns)

Measured on the byte-identical delivered archive (sha256 8fb6820f...),
extracted clean, 2 vCPU / 2 GB RAM / Python 3.10 Linux sandbox:

| check | result |
|---|---|
| first clean run, wall time | 42.0 s (venv + pip install + ingest; ingest alone 9.8 s) |
| peak child RSS during first run | 485 MB |
| installed footprint | repo 24 MB + .venv 503 MB + .index 4 MB = ~530 MB |
| pip cache on a default box | not written here (sandbox forces PIP_NO_CACHE_DIR); reproduce.sh now pins `--no-cache-dir` so behavior does not depend on box config |
| CORPUS_DIR swap (amended manifest) | rebuild, new hash, 10.1 s; swap back rebuilds again - no stale index either direction |
| rerun with dead PyPI index, deps installed | OK (pip short-circuits on satisfied pins), corpus-unchanged skip |
| unit tests | 4/4 OK |

Judge contract re-verified against the repo: freshness_contract.md (stop,
swap CORPUS_DIR, restart with the same script; stale index must not answer
from removed bytes) and EVALUATION_PROTOCOL.md (isolated container, metered
egress). Remaining network assumption: FIRST setup needs PyPI unless the
optional wheelhouse is built for the target CPython.

## 2026-10-18 — Stage 2 retrieval: fusion sweep (dev split only, 146 questions)

Candidates evaluated on role/temporal-eligible candidates, gold-section
recall of the 119 answerable dev questions:

| fusion | @1 | @3 | @6 | @10 |
|---|---|---|---|---|
| vector only | 0.445 | 0.698 | 0.798 | 0.857 |
| BM25 only | 0.656 | 0.832 | 0.924 | 0.975 |
| RRF k=60 | 0.538 | 0.790 | 0.891 | 0.933 |
| weighted sum 0.5/0.5 | 0.664 | 0.832 | 0.924 | 0.950 |
| **weighted sum 0.3/0.7 (chosen)** | **0.664** | **0.832** | **0.933** | **0.966** |

Temporal slice recall@6: 19/19. Contradiction slice recall@6: 8/8. Role
isolation: 0 violations across all 146 questions (asserted per question).
Blocked-evidence signal: fires on 21/21 role-blocked questions after moving
to a common-scale comparison. 4 answered questions miss at @10 - genuine
retrieval misses (paraphrase-heavy), candidates for the reranker stage.

Refusal-threshold groundwork: the max-normalized fused score is NOT a
usable out-of-corpus signal (normalization forces a high top score;
the 6 dev out-of-corpus questions score 0.80-0.95 fused). Raw signals are
emitted instead: best allowed-pool cosine - answered min 0.503 / median
0.704, out-of-corpus median 0.532 (n=6), role-blocked median 0.566.
Distributions overlap on dev; no threshold committed at this stage. The
service stage will tune the refusal policy on dev with this evidence.

Stage-2 unit tests: 6 new (role isolation x3 incl. blocked signal, temporal
exclusion, supersession both directions, lexical+semantic queries) -
10/10 pass.

## 2026-10-18 — Stage 3: /ask service, refusal policy, extractive answers (dev, deterministic)

Environment as before (2 vCPU / 2 GB RAM, Python 3.10, CPU-only). Vendored
grader hash-verified against RELEASE_MANIFEST.json (season-1-2026-09-07).
All numbers are the DEV split; the held-out set was never read.

Refusal policy tuning evidence (raw signals per dev question):
- blocked-evidence scan depth: k=6 misses a role-blocked question
  (indiafinbench_TMP_013) that k=10 catches; answers still use top-6.
- not_permitted rule (blocked AND allowed-pool bm25 < 23.5): 21/21 caught,
  0/119 answered refused. Margin: blocked max bm25 23.34 vs answered+blocked
  min 24.19 - under 1 BM25 point, thin.
- not_in_corpus rule (bm25 < 21 AND cosine < 0.64): 6/6 OOC caught, 2/119
  answered refused (REG_111, REG_197). OOC n=6 on dev; the rule keys on
  question-length-sensitive BM25, so short paraphrases of answerable
  questions also refuse (observed live). Refusal-biased is the intended
  product posture, but held-out OOC (n=16) is the real test.

Deterministic dev results (grader --no-judge):
| metric | value | bar |
|---|---|---|
| schema valid | 1.0 | - |
| coverage | 0.9832 | >= 0.87 |
| citation verbatim | 1.0 | >= 0.92 |
| OOC refusal | 1.0 (0 fabrications) | >= 0.95 |
| role-blocked refusal | 1.0 | >= 0.87 |
| role violations | 0 | 0 |
| /ask latency | p50 6.9 ms / p95 10.0 ms in-process; HTTP 24 req @ conc 12 in 1.2 s | p95 <= 10.8 s |
| cost | 0 API tokens | <= 1005/743 tokens |

Citation-choice experiment (gold-section hit rate on the 117 answered):
pure sentence-idf 0.521, pure top-chunk 0.667, 60/40 blend 0.667 (chosen;
tie-break rationale in DECISIONS.md #14).

UNMEASURED (no key at $0): answer correctness and citation support (rubric
judge = LLM), semantic role review, freshness pack (private). Manual
inspection of 8 sampled answers: several pick the wrong passage (extractive
ceiling ~0.67 citation choice), some right-passage answers omit the direct
conclusion - correctness below the 0.88 bar is expected for the extractive
baseline and must not be claimed otherwise.

Service lifecycle: PORT + healthz verified; bad role/body -> 400, unknown
path -> 404, no internals in error bodies; CORPUS_DIR amendment (one doc's
fraction changed) + restart -> rebuild (new corpus hash, skipped=false) and
/ask answers from amended bytes; swap-back rebuilds again.

Stage-3 unit tests: 16 new (refusal policy x2, verbatim quote, role guard x2,
temporal current/historical, refusal text hygiene, HTTP contract x7, LLM
backend stub x2) - 26/26 pass.

## 2026-10-19 — Stage 4: model generation (v1, SUPERSEDED) + spec rebuild (v2)

User steering 2026-10-19 23:35 IST: a pasted 7-step approach became the
strict spec (RRF fusion, no vector DB, role split, temporal eligibility,
temp-0 budget model with exact-quote JSON, LLM-free quote verification,
dev-only tuning). v1 (Chroma + weighted fusion 0.3/0.7) results below are
kept as the labeled superseded baseline, never relabeled as current.

v1 (SUPERSEDED architecture: Chroma store + weighted-sum fusion):
gemini-3.5-flash-lite generator, 146/146 dev questions, zero extractive
fallbacks, no quota stops. Deterministic grader: coverage 0.9832, citation
verbatim 1.0, OOC refusal 1.0 (0 fabrications), role-blocked refusal 1.0,
role violations 0. Cost at pinned list price $0.000627/question ($0.0915
total; actual spend $0, free tier). Latency p50 1.31 s / p95 6.46 s.
Raw: results/answers_dev_llm_v1_weighted_chroma_SUPERSEDED.jsonl +
grader_dev_llm_v1_weighted_chroma_SUPERSEDED.{report,run}.json.

Model availability finding (live API error, verbatim): gemini-3.8-flash -
the user's first-choice model - has a 20 requests/day free-tier cap on this
project ("Quota exceeded for metric: generate_content_free_tier_requests,
limit: 20, model: gemini-3.8-flash"), so a 146-question measurement cannot
run on it in one day; per her explicit "if not use lower models too" the
generator is gemini-3.5-flash-lite (budget class per MODELS.md Flash tier;
live-observed project limits 15 RPM / 250K TPM / 500 RPD; list $0.30/$2.50
per 1M, https://ai.google.dev/gemini-api/docs/pricing). The 3.8-flash
smoke answer (1 question) was correct with a verbatim citation.

v2 (current architecture: npz/json store + RRF k=60): retrieval dev
recall@1/3/6/10 = 0.5378/0.7899/0.8908/0.9328, 0 role-leak violations,
temporal slice recall@6 1.0 (n=19), contradiction slice 1.0 (n=8), full
146-question scan 0.78 s CPU. Ingest: 396 chunks, embed 9.6 s, index write
0.028 s, .index 1.5 MB. Generation + judged results: PENDING (run in
progress at log time; filled below when measured).

UNMEASURED still: official pinned judge (gpt-4.1-mini-2025-04-14) - dev
judged metrics will use substitute judge gemini-3.5-flash, labeled;
semantic role review; private freshness pack.

### v2 generation run (2026-10-19 23:52 -> 2026-10-20 01:12 IST)

Config: OP05_GEN_BACKEND=openai, OP05_GEN_MODEL=gemini-3.5-flash-lite (temp-0
JSON, OP05_GEN_STRICT=1), RRF k=60 retrieval, no vector DB, 4s pacing.

Results (146/146 rows, deterministic gates; judged metrics pending):
schema 1.0 | citation verbatim 1.0 | OOC refusal 6/6 | role refusals 21/21 |
role violations 0 | fabrications 0 | answered-on-answerable 117/119 (0.9832,
2 false refusals - both zero-generation-spend extractive-path refusals).

Generation accounting: 90 successful LLM API calls; 88,753 input / 23,316
output tokens (avg 758.6/199.3 per LLM answer, inside the 1005/743
reference bar); $0.0849 list price, $0 actual (free tier).
28 of the 117 answered rows took the extractive-fallback path after
transient endpoint failures (STRICT aborts on quota, falls back on
transient). Per-row provenance was not recorded by the code that ran;
answer_eval now writes "generator"/"extractive_fallback" per row, and
results/answers_dev_llm_provenance_inferred.jsonl carries INFERRED labels
(cost==0 heuristic, aggregate matches the run summary exactly: 89 answered
rows with positive generation spend + 28 with zero; 90 API calls vs 89
positive-spend answered rows leaves one row that consumed a call without an
LLM answer - spent-call fallback or spent-call refusal; unresolvable from
the raw rows, hence the new labels). Raw answers_dev_llm.jsonl was briefly modified post-run (two label keys
added) and then reconstructed by removing them: content (answers, citations,
costs, latencies) is untouched, but NO pre-modification hash or copy exists,
so byte-identity with the run's original bytes is plausible, not proven.
Recorded honestly: the file is a reconstructed original-format dataset, not
verified raw evidence. The labeled variant lives at
results/answers_dev_llm_provenance_inferred.jsonl.

Timing caveat (material): wall launch->finish was 79m56s, but the process's
own monotonic loop timer measured 962.83s. The ~64-minute gap is CONSISTENT WITH a
sandbox infrastructure suspension (plausible explanation, not a verified
platform diagnosis): the workspace shows uptime/boot evidence of a clock
discontinuity ending ~00:14 IST while the same PID and log fd predate it
(23:52); CLOCK_MONOTONIC excludes suspend time where filesystem timestamps
include it. No owning-source confirmation of a suspension exists. Consequences: (a) per-row monotonic latencies are
the honest numbers: p50 1199ms, p90 2158ms, p95 3829ms, p99 13.6s, max
90.1s (one row hit the 90s socket timeout); (b) batch wall time tonight is
polluted by the suspension and says nothing about the endpoint; (c)
earlier in-night observations of "multi-minute endpoint stalls" cannot be
cleanly attributed between the endpoint and the suspension - withdrawn as
evidence. No latency-readiness claim either way.

Transport hardening added AFTER this run (post-measurement; future runs
only, config distinction preserved): whole-request deadline - one
wall-clock budget covering connect/request/headers/body per attempt
(_read_bounded trickle cap + remaining-budget handoff), and one total
deadline across all 4 retries incl. backoff sleeps (~6.5 min worst case);
row-level checkpointing (.checkpoint.jsonl per completed row). Stub-tested,
33/33 pass. v2 was measured WITHOUT these; re-measurement under the new
transport is a separate future run decision (quota cost: 146 calls).
