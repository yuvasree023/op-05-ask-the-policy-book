# OP-05 — Ask the Policy Book

Solution for the Deployment.inc Season 1 hiring problem
[OP-05](https://github.com/Deployment-inc/Deployment.inc-Hiring-Problems/blob/main/problems/OP-05-ask-the-policy-book.md):
a compliance assistant that answers from a shipped document corpus only, with
verbatim citations, role-based isolation and temporal (as_of) correctness.

**Current state: stage 3 — ingestion + retrieval + the `/ask` service with a
deterministic extractive generator and refusal policy.** Remaining: the
LLM-judged metrics (answer correctness, citation support) are unmeasured -
they need a budget-class judge key - and the analysis docs (MEMO.md,
LANDSCAPE.md) land with final submission. Everything is evaluated on the dev
split only; the held-out set is never touched.

## Stage 1: what exists

`CORPUS_DIR -> validate -> chunk -> embed -> local vector DB`

- Reads the published corpus format (`manifest.jsonl`, `passages.jsonl`,
  `text/<doc_id>.txt` with `[doc_id#sNNNNN]` section markers).
- Validates every document's sha256 against the manifest; rejects tampered or
  malformed corpora with a clear error.
- Chunks on section markers (citation `section` ids preserved), splitting only
  over-long sections at paragraph/sentence boundaries; every chunk stays an
  exact substring of the source bytes so citation quotes can verify verbatim.
- Embeds with a vendored, quantized all-MiniLM-L6-v2 (ONNX, CPU, no download
  at run time — the review sandbox restricts egress).
- Stores chunks + embeddings as one local npz/json cache pair (no vector
  database anywhere: 391 passages score brute-force) with the metadata later
  stages filter on: `role`, `effective_date`, `supersedes`, `doc_sha256`,
  `section`, `part`.
- Corpus-hash keyed index: an unchanged corpus skips re-embedding; a swapped
  or amended corpus (the freshness update pack path) rebuilds automatically.

## Stage 2: hybrid retrieval (role + as_of aware)

`question + role + as_of -> ranked eligible chunks`

- Hybrid scoring: max-normalized weighted sum of BM25 (in-package, stdlib
  only) and vendored-MiniLM cosine. Weights (0.3 vector / 0.7 lexical) tuned
  on the dev split only; sweep in EXPERIMENT_LOG.md.
- **Role isolation by construction:** both channels are computed only over
  chunks whose document the requester's role may see (`public` < `ops` <
  `legal`, the grader's ROLE_RANK). Above-role documents never enter the
  candidate pool, so they cannot leak into evidence. A separate boolean
  `blocked_evidence` signal records that matching text exists above the
  role - the difference between refusing `not_permitted` and falsely
  claiming `not_in_corpus`; the denied text itself never leaves the signal.
- **Temporal eligibility:** a document is eligible at `as_of` when
  `effective_date <= as_of` and it is not superseded by a corpus document
  whose own `effective_date <= as_of` (the freshness update pack's
  mechanism; a historical `as_of` still reaches the older version).
- Raw per-channel signals (`top_cosine`, `top_bm25`) are emitted for the
  refusal-threshold work in the service stage.

## Stage 3: the /ask service

`question + role + as_of -> {id, status, answer, citations}`

- **Service contract (freshness_contract.md):** `POST /ask` on `$PORT`,
  `GET /healthz` 200 once the index is loaded. stdlib HTTP only - no new
  dependencies, the stage 1-2 footprint and offline guarantees are unchanged.
- **Refusal policy (dev-tuned, margins in EXPERIMENT_LOG.md):** blocked
  evidence (a denied document would have ranked top-10) + a weak allowed
  pool -> `not_permitted` ("access denied", names no document, quotes
  nothing); both retrieval channels weak -> `not_in_corpus`. Refusals carry
  no citations.
- **Extractive generator (default, $0):** the answer quotes the strongest
  eligible evidence window verbatim with source attribution and effective
  dates; future-dated supersessions are disclosed in the answer. Every quote
  is re-verified as an exact substring of the cited passage before returning,
  and the answer is checked for any 12-token overlap with above-role
  documents (the grader's role check). Zero API tokens, millisecond latency.
- **Optional LLM generator** (`policybook/llm.py`, OFF by default): a
  budget-class OpenAI-compatible backend with quote verification, one retry,
  drop-or-refuse. Never called unless `OP05_GEN_BACKEND=openai` and a key
  are explicitly configured; not required to run any part of this stage.

## Run it

```bash
./scripts/reproduce.sh                     # ingest, then serve /ask on $PORT (default 8000)
./scripts/reproduce.sh --selfcheck         # offline dev evaluation, then exit
./scripts/reproduce.sh --force             # rebuild the index even though nothing changed
CORPUS_DIR=/path/to/other-corpus ./scripts/reproduce.sh   # any corpus dir in the published format
RETRIEVAL_EVAL=0 ./scripts/reproduce.sh --selfcheck       # skip the retrieval recall report
JUDGE=1 LLM_BASE_URL=... LLM_API_KEY=... JUDGE_MODEL=... ./scripts/reproduce.sh --selfcheck  # + rubric judge
```

`CORPUS_DIR` defaults to `./data/corpus` (included, see below), `INDEX_DIR`
to `./.index`. Requirements: Python 3.10+ with `venv`, Linux, no GPU. The
script creates `.venv`, installs pinned requirements and runs ingestion;
indexing time is reported separately in the JSON summary and in
`results/ingest_report.json`.

**Bundled corpus:** `data/corpus/` is the published OP-05 corpus (IndiaFinBench,
CC BY 4.0, redistributable; provenance in `data/corpus/manifest.json`). It is
an input, not part of the solution: `CORPUS_DIR` swaps it, which is exactly
what the freshness update pack does at review time.

**Offline scope:** the embedding model (`models/`), corpus and dev questions
ship in the repo, so ingestion and retrieval run fully offline. The only
network touchpoint is pip installing pinned requirements on first run;
`scripts/vendor_wheels.sh` builds a local wheelhouse (`vendor/wheels/`)
before submission so the reviewer's egress-restricted container needs no
network at all. pip runs with `--no-cache-dir`, so no wheel cache (hundreds
of MB) is written on boxes that do not already disable it. No telemetry
anywhere - the stack is stdlib + numpy + onnxruntime + sentencepiece.

## Measured requirements (2 vCPU / 2 GB RAM / Python 3.10 Linux sandbox)

| what | measured |
|---|---|
| repo working tree (model + corpus + code) | 24 MB |
| `.venv` after `pip install` | MEASURE-IN-PACKAGING (Chroma removed 2026-10-19) |
| `.index` after ingesting the shipped corpus | 1.5 MB |
| **total installed footprint** | **MEASURE-IN-PACKAGING; keep ~1 GB free** |
| peak RAM during ingestion | 485 MB |
| first clean run (venv + install + ingest) | ~45 s |
| /ask latency (dev, 146 questions) | p50 ~7 ms, p95 ~10 ms, 0 API tokens |
| unchanged-corpus rerun | skip, < 2 s |
| corpus swap via CORPUS_DIR | full rebuild, ~10 s |

No GPU is used anywhere; everything above is CPU-only.

## Layout

```
policybook/            solution package
  corpus.py            corpus reader + format/sha256 validation + corpus hash
  chunk.py             section-marker chunking (exact substrings only)
  embed.py             vendored ONNX MiniLM encoder
  db.py                npz/json passage store, corpus-hash rebuild logic (no vector DB)
  ingest.py            ingestion CLI entry point
  bm25.py              Okapi BM25 (stdlib only)
  retrieve.py          hybrid retrieval: role filter, as_of eligibility, RRF fusion
  retrieve_eval.py     dev-split retrieval evaluation CLI
  answer.py            refusal policy + extractive answer composition + role guard
  service.py           POST /ask + GET /healthz HTTP service (stdlib)
  llm.py               optional budget-class LLM generator (OFF by default)
  answer_eval.py       dev answers + vendored-grader self-check CLI
data/corpus/           shipped OP-05 corpus (input, swappable via CORPUS_DIR)
data/questions_dev.jsonl   published dev questions (tuning evidence only)
models/all-MiniLM-L6-v2/   vendored quantized embedding model
scripts/reproduce.sh   one-command reproduction (serve, or --selfcheck to evaluate)
tools/op05_grader/     vendored published grader + rubric (hash-verified, self-check only)
tests/                 synthetic-corpus unit tests (26 tests)
results/               ingest/retrieval reports, answers_dev.jsonl, grader_dev_report.json
```

## Tests

```bash
python3 -m venv .venv && ./.venv/bin/pip install -q -r requirements.txt
./.venv/bin/python -m unittest discover -s tests    # 26 tests
```

## Stage 3 measured (dev split, 146 questions, deterministic checks via the
vendored published grader - no LLM judge)

| bar | target | measured (dev) |
|---|---|---|
| schema valid | - | 1.0 |
| coverage (answered/answerable) | >= 0.87 | **0.9832** (117/119; 2 false refusals) |
| citation verbatim rate | >= 0.92 | **1.0** |
| out-of-corpus refusal | >= 0.95, 0 fabrications | **1.0** (6/6), 0 |
| role-blocked correct refusal | >= 0.87 | **1.0** (21/21) |
| role violations | 0 | **0** |
| answer correctness / citation support | >= 0.88 / >= 0.82 | **unmeasured** - needs the budget-class rubric judge (LLM key); the extractive baseline is expected BELOW the correctness bar |
| cost per question | <= 1005 in / 743 out tokens | **0 tokens** (extractive) |
| latency | p95 <= 10.8 s @ conc 12 | **~10 ms** in-process; HTTP service 24 requests @ conc 12 in ~1.2 s |

Service lifecycle tested: PORT/healthz contract, 400s on bad input, CORPUS_DIR
swap + restart rebuilds and answers from the amended bytes, no stale index.

Stage-1 measured on the shipped corpus (2 vCPU sandbox): 34 documents, 391
sections, 396 chunks (5 sections split), zero validation warnings, embed
~10 s, index 0.3 s; unchanged-corpus rerun skips in ~0.1 s.

Stage-2 measured on the 146 published dev questions: retrieval recall
@1 0.664 / @3 0.832 / @6 0.933 / @10 0.966; temporal and contradiction
slices 100% at k=6; zero role-isolation violations across all 146 questions
(including the 21 role-blocked ones); blocked-evidence signal fires on
21/21 role-blocked questions. All 146 queries run in ~1 s total, CPU-only.
Detail: `results/retrieval_report.json` and `results/retrieval_dev.jsonl`.

## Submission access requirements (grounded 2026-10-19)

Source: Deployment-inc/Deployment.inc-Hiring-Problems README.md, CONTACT.md, SUBMISSION_SCHEMA.md (live repo).

- Repository: a separate **private** GitHub repository (never the Fork button; forks of a public repository are public).
- Reviewer access: invite only the GitHub account **aayush-deployment-inc** (https://github.com/aayush-deployment-inc). Share repository and artifact access only with this account unless Deployment.inc names another reviewer privately. Candidates are never added to a shared submission repository.
- Submission route: private form at https://deployment.inc/hiring-challenges/#submit with name, email, phone, optional LinkedIn, short introduction, chosen problem, private repository URL and the exact 40-character commit SHA, plus `submission.yaml` per SUBMISSION_SCHEMA.md (problem, repository and commit must match the form).
- After saving, the form returns a submission ID (delivery receipt only). A reviewer confirms repository access by email within ten working days (IST).
- No public forks, issues, discussions or PRs carrying submission work, results, contact details or credentials. Keep submission-specific code, methods and results private through the season (closes 7 December 2026, 23:59 IST).
- A private evaluation-only licence is sufficient; the work stays the author's. Raw benchmark outputs belong in `results/raw/` (every run, including failed ones); outputs over 100 MB go in an access-controlled private release shared only with reviewers, with SHA-256 in `results/manifest.json`. An unlisted or hard-to-guess public URL is not private.
- Resubmission: new form entry with updated YAML and SHA, then email old and new submission IDs to contact@deployment.inc. Withdraw any time via the same address with the submission ID.
