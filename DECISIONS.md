# DECISIONS

Durable choices, newest last. Each entry: what, and why.

## 2026-10-18 — Stage 1 (ingestion)

1. **Vendored quantized ONNX all-MiniLM-L6-v2 instead of a pip-downloaded
   sentence-transformers model.** The reviewer's reproduction container
   restricts egress to a metered gateway (EVALUATION_PROTOCOL.md), so no
   Hugging Face download can happen at reproduce time; the model ships in
   `models/` (~23 MB quantized). Quantized ONNX also removes the ~200 MB
   torch dependency from a CPU-only box. The approved approach named
   "bge-small or MiniLM"; MiniLM won on artifact size. If retrieval quality
   suffers on dev, bge-small-en-v1.5 ONNX (~34 MB) is the swap-in candidate.
2. **Chroma as the vector DB**, local and persistent, because stage 2+ needs
   metadata filtering (`role`, `effective_date`, `supersedes`) for role
   isolation and temporal filtering, and Chroma filters server-side on
   stored metadata. Brute force over 391 passages would also work; the DB
   keeps the door open for larger corpora and costs nothing locally.
3. **Chunk on the corpus's own `[doc_id#sNNNNN]` section markers**, splitting
   only sections over ~1000 chars at paragraph/sentence boundaries. Citations
   name `section` ids, so chunking must not invent new boundaries that lose
   them. All chunk text is exact source substring (offset-based splitting),
   never re-wrapped text, because the grader checks quotes verbatim (NFKC,
   soft-hyphen strip, whitespace collapse, case fold) against the text files.
4. **Corpus-hash keyed index.** sha256 over sorted manifest lines plus the
   passages.jsonl digest, stored in collection metadata. Identical corpus
   skips re-embedding; any byte or metadata change (the freshness update
   pack path: stop, swap CORPUS_DIR, restart) forces a full rebuild. Verified
   with a metadata-only amendment test.
5. **Text files are authoritative; passages.jsonl is a cross-check.** The
   verbatim quote checker reads `text/<doc_id>.txt`, so sections are parsed
   from those bytes and passages.jsonl only validates the parse (id set and
   text equality, reported in `validation`).
6. **No generic folder fallback.** The published corpus format
   (manifest.jsonl + passages.jsonl + text/) is the only accepted input;
   anything else fails loudly. Generic parsing would invent metadata (roles,
   dates) the role-isolation bars depend on.

### Open items for later stages

- **Quote repair policy (stage 6):** the earlier approach note said "snap to
  the nearest sentence" for bad quotes. Snapping by text similarity can
  detach a quote from the answer it supports, which risks the 0.82 support
  bar. Current intent: verify each quote as a normalized substring of the
  cited passage; on failure retry generation once, then drop the citation
  (and refuse if nothing supportable remains). Revisit with dev-set evidence.
- ~~requirements.txt pins pip packages but pip itself needs egress~~
  **Resolved 2026-10-18:** `scripts/vendor_wheels.sh` builds
  `vendor/wheels/` (91 wheels, ~102 MB) and reproduce.sh auto-detects it,
  installing `--no-index`. Tested: fresh venv install from the wheelhouse
  with PyPI unreachable, then full ingestion, all offline. Wheelhouse is a
  submission-time artifact (too large for chat handoffs); build it on the
  same OS/CPython as the review target before publishing the private repo.
- **No role leakage at ingestion:** all roles' chunks share one collection;
  isolation is enforced at retrieval/generation (stage 2+), never by
  embedding restricted text into prompts. Recorded here so the design
  constraint is visible from stage 1.

## 2026-10-18 — Stage 2 (retrieval) + footprint hardening

7. **Hybrid retrieval: max-normalized weighted sum of BM25 and vendored
   MiniLM cosine (0.3 vector / 0.7 lexical).** Both channels score the same
   role/temporal-eligible candidate set; fusion happens after filtering,
   never before. BM25 is implemented in-package (stdlib only, ~60 lines)
   because the corpus is ~400 chunks: a `rank_bm25` dependency would add
   install weight for nothing. The weight split was tuned on the dev split
   only (sweep in EXPERIMENT_LOG.md); reciprocal-rank fusion was tried first
   and lost at every k. The vector channel is kept despite being weaker on
   dev because BM25 alone has no paraphrase robustness, which the hidden
   set will exercise.
8. **Role isolation by construction, not by post-filter.** The candidate
   pool for both channels is built from `ROLE_RANK` (the grader's own
   public < ops < legal ordering) before any scoring. Above-role chunks are
   never scored as evidence, so there is nothing to leak into a later
   prompt or citation. The `blocked_evidence` boolean (would a denied chunk
   have made top-k on a common scale?) is computed separately and carries
   no text - it exists so the service stage can refuse `not_permitted`
   instead of lying `not_in_corpus` (the exact failure the reference
   pipeline was called out for).
9. **Temporal eligibility = effective_date <= as_of AND not superseded by a
   document itself effective by as_of.** Supersession is read from the
   manifest's `supersedes` field, the freshness update pack's documented
   metadata channel. Verified: zero dev gold citations have an
   effective_date after their as_of, so the date filter cannot exclude a
   legitimate answer on the published data. A historical as_of still
   reaches the superseded version (the contract allows historical citation;
   it prohibits presenting a superseded source as *current*).
10. **In-process scoring over Chroma where-filters.** At 391 passages the
    whole index (396 x 384 floats) loads in milliseconds; both channels then
    share one identical candidate set with no filter-semantics drift between
    stores. Chroma remains the persistent, corpus-hash-keyed store from
    stage 1. If a future corpus is orders of magnitude larger this moves
    back into the DB - noted, not built.
11. **Footprint hardening from the reproduction audit.** pip install now
    uses `--no-cache-dir` (a default box would otherwise keep a several
    hundred MB wheel cache) and Chroma telemetry (posthog) is disabled in
    code - "offline" must mean no traffic at all, not just no downloads.
    Measured totals recorded in README: ~530 MB installed, 485 MB peak RAM,
    ~45 s first run. The 503 MB `.venv` is mostly Chroma's own dependency
    chain (kubernetes, sympy, grpc, ...); pruning site-packages after
    install was rejected as too fragile to defend in review.

## 2026-10-18 — Stage 3 (/ask service, refusal policy, extractive generator)

12. **Default generator is deterministic extractive composition, not an LLM
    call.** The user constraint is $0 with no billing card, and the review
    sandbox is CPU-only with metered egress. The answer is composed from the
    strongest eligible evidence window, quoted verbatim with source
    attribution and effective dates; future-dated supersessions are
    disclosed. Consequences, stated plainly: every citation quote verifies
    against the cited passage (measured 1.0 verbatim on dev) and role safety
    is checkable (measured 0 violations), but ANSWER CORRECTNESS IS NOT
    CLAIMED - right passage selection is ~0.67 on dev and extraction can
    miss qualifications, so the 0.88 correctness bar almost certainly needs
    the budget-class LLM generator. That is a later stage and a real model
    decision, not a threshold tweak. The optional backend (policybook/llm.py)
    is plumbed (OpenAI-compatible, no temperature field,
    max_completion_tokens, quote verification + one retry + drop/refuse,
    usage accounting) and stub-tested offline; it is OFF unless
    OP05_GEN_BACKEND=openai and a key are explicitly configured.
13. **Refusal thresholds from raw channel signals, not fused scores.** The
    max-normalized fused score cannot detect out-of-corpus questions (stage
    2 finding). Policy: blocked-evidence (scanned at k=10 while answers use
    top-6 - the TMP_013 miss at k=6 proved the shallower scan insufficient)
    plus allowed-pool BM25 < 23.5 -> not_permitted; BM25 < 21 AND cosine <
    0.64 -> not_in_corpus. Dev margins are THIN (not_permitted max 23.34 vs
    answered+blocked min 24.19; OOC n=6) and recorded in EXPERIMENT_LOG.md;
    the policy errs toward refusing, which a compliance tool should.
14. **Citation choice blends sentence match with retrieval rank (60/40).**
    Pure sentence-idf choice picked a gold section on 52% of dev answers,
    pure top-chunk 66.7% (= recall@1 ceiling), the blend ties at 66.7% and
    can recover rank-1 misses - tie broken for robustness, recorded honestly.
15. **The service is stdlib http.server, threaded.** No new dependency:
    installed footprint stays ~530 MB, offline guarantee untouched. Error
    responses are generic JSON - a stack trace or echo can leak restricted
    text as surely as an answer can. Role strings outside the published set
    are rejected 400; role text inside the question is never authoritative.
16. **Grader vendored, hash-verified.** tools/op05_grader/{grader.py,
    rubric.md} match RELEASE_MANIFEST.json season-1-2026-09-07 pins
    (sha256 recorded in tools/op05_grader/PROVENANCE.md), so the dev
    self-check runs the exact deterministic checks the reviewer recomputes,
    offline. The rubric judge is optional and off without a key.
