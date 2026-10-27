# OP-05 - Ask the Policy Book

A compliance assistant that answers questions over a shipped policy corpus,
and nothing else. Every answer carries verbatim citations (document,
section, exact quote), respects three access levels (public < ops < legal),
and is correct as of the question's date, including superseded documents.
Built as a solution for the Deployment.inc Season 1 hiring problem
[OP-05](https://github.com/Deployment-inc/Deployment.inc-Hiring-Problems/blob/main/problems/OP-05-ask-the-policy-book.md).

The system says "no" in two distinct ways, on purpose:

- `not_in_corpus`: the corpus does not contain the answer. It says so and
  guesses nothing.
- `not_permitted`: a matching document exists above the requester's access
  level. The refusal says that access is denied without naming, quoting or
  paraphrasing the restricted document. Pretending no rule exists is the
  dangerous failure mode here: a compliance officer told there is no rule
  will act as though there is none.

## Architecture

```
CORPUS_DIR -> validate -> chunk -> embed -> index
question + role + as_of -> retrieve -> generate -> verify -> {id, status, answer, citations}
```

**Ingestion.** Reads the published corpus format (`manifest.jsonl`,
`passages.jsonl`, `text/<doc_id>.txt` with `[doc_id#sNNNNN]` section
markers) and validates every document's sha256 against the manifest;
tampered or malformed corpora are rejected with a clear error. Chunking
follows section markers so citation `section` ids survive, splitting only
over-long sections at paragraph or sentence boundaries. Every chunk stays
an exact substring of the source bytes, which is what makes verbatim
citation verification possible. Embeddings come from a vendored, quantized
all-MiniLM-L6-v2 (ONNX, CPU, no download at run time).

**Index: no vector database.** Chunks, metadata and embeddings are stored
as one npz/json pair holding `role`, `effective_date`, `supersedes`,
`doc_sha256`, `section` and `part` per chunk. 396 chunks score brute-force
in numpy in under a second for the entire 146-question dev set, so an ANN
index would buy nothing. The index is keyed by corpus content hash: an
unchanged corpus skips re-embedding, a swapped or amended corpus (the
freshness update-pack path) rebuilds automatically.

**Hybrid retrieval with reciprocal rank fusion.** Two channels, BM25
(stdlib-only Okapi implementation) and MiniLM cosine similarity, are fused
with RRF (k=60, Cormack et al. 2009). Rank fusion removed the tuned
0.3/0.7 channel weights of the first iteration: one less dev-tuned
parameter to overfit.

**Role isolation by construction.** Above-role documents are gated out
before anything reaches the generator, so restricted text cannot leak into
an answer. A separate `blocked_evidence` signal records that matching text
exists above the role (a denied document would have ranked top-10) without
exposing its content. That signal is the difference between a correct
`not_permitted` and a false `not_in_corpus`. A 12-token overlap check runs
on every generated answer against above-role documents as a second,
independent guard.

**Temporal eligibility.** A document is eligible at `as_of` when
`effective_date <= as_of` and it is not superseded by a corpus document
whose own `effective_date <= as_of`. A historical `as_of` still reaches
the older version of a rule; answers disclose future-dated supersessions
when one is already in the corpus.

**Answer generation.** Two backends, selected explicitly or by
environment:

- **Model generator** (`policybook/llm.py`): a budget-class
  OpenAI-compatible backend. It activates automatically when
  reviewer-gateway credentials (`LLM_BASE_URL` + `LLM_API_KEY`) are
  present; `LLM_MODEL` is optional, and when omitted the request names no
  model so the gateway serves its default. `OP05_GEN_BACKEND` overrides
  selection explicitly.
- **Extractive composer** (default, zero API tokens): quotes the strongest
  eligible evidence window verbatim with source attribution and effective
  dates. When it runs, that is visible everywhere: startup banner,
  `X-Generator` response header, per-row generator field in batch outputs.

On the live `/ask` path the model call is one bounded attempt under a
whole-request deadline; on timeout, stall or provider 503 the service
returns `503 + Retry-After`. It never silently falls back to the
extractive composer: a failed answer is a visible failure, not hidden
degradation. For development and scoring runs, an optional second key
(`LLM_API_KEY_FALLBACK`) fails over on a daily-quota 429 - same model,
same endpoint, key-level only, and the switch is recorded per run.

**Citation verification.** Every quote must be a verbatim substring of the
cited passage after normalization. Failures retry generation once, then
the citation is dropped or the answer refused. Quotes are never snapped to
a nearby sentence: substring match proves copying, and a snapped quote can
attach evidence that does not support the answer.

**Refusal policy.** Blocked evidence plus a weak allowed pool yields
`not_permitted` (names no document, quotes nothing); both retrieval
channels weak yields `not_in_corpus`. Refusals carry no citations.
Thresholds were tuned on the dev split only, margins in EXPERIMENT_LOG.md.

## Run it

```bash
./scripts/reproduce.sh                     # ingest, then serve POST /ask on $PORT (default 8000)
./scripts/reproduce.sh --selfcheck         # offline dev evaluation, then exit
./scripts/reproduce.sh --force             # rebuild the index even though nothing changed
CORPUS_DIR=/path/to/other-corpus ./scripts/reproduce.sh   # any corpus in the published format
RETRIEVAL_EVAL=0 ./scripts/reproduce.sh --selfcheck       # skip the retrieval recall report
JUDGE=1 LLM_BASE_URL=... LLM_API_KEY=... JUDGE_MODEL=... ./scripts/reproduce.sh --selfcheck  # + rubric judge
```

`CORPUS_DIR` defaults to `./data/corpus` (bundled, see below), `INDEX_DIR`
to `./.index`. Requirements: Python 3.10+ with `venv`, Linux, CPU only
(pinned dependencies: numpy, onnxruntime, tokenizers). The script creates
`.venv`, installs pinned requirements, runs ingestion, then serves
`POST /ask` with `GET /healthz` for readiness.

**Bundled corpus:** `data/corpus/` is the published OP-05 corpus
(IndiaFinBench, CC BY 4.0, redistributable; provenance in
`data/corpus/manifest.json`). It is an input, not part of the solution:
`CORPUS_DIR` swaps it, which is exactly what the freshness update pack
does at review time.

**Offline scope:** the embedding model (`models/`), corpus and dev
questions ship in the repo, so ingestion and retrieval run fully offline.
The only network touchpoint is pip installing pinned requirements on first
run; `scripts/vendor_wheels.sh` builds a local wheelhouse
(`vendor/wheels/`) so an egress-restricted review container needs no
network at all. No telemetry anywhere.

## Evaluation

Two kinds of measurement, kept strictly separate:

- **Deterministic gates**: schema validity, verbatim citation rate,
  coverage, refusal correctness, role violations, fabrication counts. These
  need no judge and are exact.
- **Judged quality metrics** (answer accuracy, citation support): measured
  on the dev split with a stand-in judge (gemini-3.5-flash) in place of the
  pinned judge (gpt-4.1-mini-2025-04-14), which has no $0 route. These are
  diagnostic only - evidence about answer quality, never an official score.
  The stand-in is the same model family as the dev generator
  (gemini-3.5-flash-lite), and self-family judging inflates agreement, so
  the real result under the official judge could land higher or lower.

All tuning used the 146 published dev questions only. The held-out set was
run exactly once, after tuning froze, as a dress rehearsal with
deterministic checks.

### Dev split (146 questions, vendored published grader 2.0)

| bar | target | measured |
|---|---|---|
| schema valid | - | 1.0 |
| coverage (answered/answerable) | >= 0.87 | 0.9832 (117/119; 2 false refusals) |
| citation verbatim rate | >= 0.92 | 1.0 |
| out-of-corpus refusal | >= 0.95, 0 fabrications | 1.0 (6/6), 0 fabricated |
| role-blocked correct refusal | >= 0.87 | 1.0 (21/21) |
| role violations | 0 | 0 |
| answer accuracy | >= 0.88 | 0.8403 (substitute judge, diagnostic) |
| citation support | >= 0.82 | 0.7037 (substitute judge, diagnostic) |
| temporal slice accuracy | - | 0.9474 (substitute judge, n=19) |
| contradiction slice accuracy | - | 1.0 (substitute judge, n=8) |

judge_errors = 0 across 253 evaluations; fabrication review complete with
0 fabricated. Full history, including judge request-shape fixes and quota
handling, is in EXPERIMENT_LOG.md.

### Retrieval (dev split)

recall @1 0.538 / @3 0.790 / @6 0.891 / @10 0.933; temporal and
contradiction slices 1.0 at k=6; zero role-isolation violations across all
146 questions; the blocked-evidence signal fires on 21/21 role-blocked
questions. All 146 queries score in under 1 s total, CPU-only. Detail:
`results/retrieval_report.json`.

### Held-out dress rehearsal (272 questions, run once on 2026-10-25)

The held-out set ships without gold answers, so nothing judged is claimed:
these are structural, gold-free checks (the vendored grader itself refuses
unlabeled gold by design).

| check | result |
|---|---|
| completion | 272/272 rows, single pass, no quota abort |
| status distribution | 197 answered / 62 not_permitted / 13 not_in_corpus |
| status enum violations | 0 |
| citations | 234 total, verbatim rate 1.0 under the grader's normalized comparison (2/234 differ from raw bytes in formatting only) |
| role violations (hierarchical) | 0 |
| role guard trips | 0 |
| answered with zero citations | 0 |
| generator mix (answered rows) | 174 model / 23 extractive, per-row provenance recorded |
| per-row latency in the batch run | p50 1.27 s, p95 5.29 s |

Artifacts: `results/answers_heldout.jsonl`,
`results/heldout_structural_report.json`.

### Footprint and latency (2 vCPU / 2 GB RAM / Python 3.10 Linux sandbox)

| what | measured |
|---|---|
| repo working tree (model + corpus + code) | 26 MB |
| `.venv` after `pip install` | 241 MB |
| `.index` after ingesting the shipped corpus | 1.5 MB |
| total installed footprint | ~270 MB |
| peak RAM during ingestion | 485 MB |
| first clean run (venv + install + ingest) | ~45 s |
| /ask latency, extractive default (48 reqs @ concurrency 12) | p50 30 ms, p95 1010 ms, 0 API tokens |
| unchanged-corpus rerun | skip, < 2 s |
| corpus swap via CORPUS_DIR | full rebuild, ~10 s |

Ingesting the shipped corpus: 34 documents, 391 sections, 396 chunks, zero
validation warnings. No GPU is used anywhere.

## Key design decisions

The short list; DECISIONS.md carries the full record with alternatives
considered.

- **No vector database.** At 396 chunks, brute-force numpy scores the whole
  dev set in under a second. Removing Chroma cut the index to 1.5 MB and
  dropped the chromadb + posthog dependency tree from the install.
- **Reciprocal rank fusion over tuned weighted-sum fusion.** RRF (k=60)
  fuses BM25 and embedding ranks with no weight parameters, removing a
  dev-tuned knob that only existed to be overfit.
- **Role exclusion before generation, with a blocked-evidence signal.**
  Restricted text never reaches the model, and the signal turns "a better
  answer exists but is withheld" into the correct generic refusal instead
  of a false "not in corpus".
- **Two refusal types, kept distinct.** `not_in_corpus` and `not_permitted`
  carry different operational meaning; collapsing them is the failure mode
  the challenge's own post-mortem describes.
- **Verbatim quotes, verified, never snapped.** A quote is an exact
  substring of the cited passage or it is dropped; evidence is never
  smoothed into place.
- **Honest degradation instead of silent fallback.** With no gateway
  credentials the extractive composer runs, labeled on every surface. On
  the live path a model stall returns 503 + Retry-After rather than an
  answer the caller cannot distinguish from a model answer.
- **Whole-request deadlines.** urllib per-operation timeouts do not cap a
  peer that trickles bytes; generation runs under one wall-clock budget
  per attempt and one total deadline across the retry chain.
- **Dev-only tuning, held-out run once.** Every threshold and prompt
  iteration was validated against the 146 dev questions; the held-out set
  was never read during development and ran a single time after the
  pipeline froze.
- **$0 end to end.** Development and evaluation ran on free-tier model
  access only, with per-row checkpointing and resume so quota boundaries
  never corrupted a measurement. No paid fallback was used.

## Layout

```
policybook/            solution package
  corpus.py            corpus reader + format/sha256 validation + corpus hash
  chunk.py             section-marker chunking (exact substrings only)
  embed.py             vendored ONNX MiniLM encoder
  db.py                npz/json passage store, corpus-hash rebuild logic
  ingest.py            ingestion CLI entry point
  bm25.py              Okapi BM25 (stdlib only)
  retrieve.py          hybrid retrieval: RRF fusion, role gate, as_of eligibility
  retrieve_eval.py     dev-split retrieval evaluation CLI
  answer.py            refusal policy + extractive composition + role guard
  service.py           POST /ask + GET /healthz HTTP service (stdlib)
  llm.py               budget-class model generator (auto-activates on gateway credentials)
  answer_eval.py       dev answers + vendored-grader self-check CLI
data/corpus/           shipped OP-05 corpus (input, swappable via CORPUS_DIR)
data/questions_dev.jsonl   published dev questions (tuning evidence only)
models/all-MiniLM-L6-v2/   vendored quantized embedding model
scripts/reproduce.sh   one-command reproduction (serve, or --selfcheck to evaluate)
tools/op05_grader/     vendored published grader + rubric (hash-verified, self-check only)
tests/                 unit tests (44)
results/               ingest/retrieval/grader reports, dev + held-out answers
DECISIONS.md           design decision log with alternatives considered
EXPERIMENT_LOG.md      measurement history, including failed and superseded runs
MEMO.md                one-page brief for a compliance head
LANDSCAPE.md           ecosystem survey and what was kept from it
```

## Tests

```bash
python3 -m venv .venv && ./.venv/bin/pip install -q -r requirements.txt
./.venv/bin/python -m unittest discover -s tests    # 44 tests
```
