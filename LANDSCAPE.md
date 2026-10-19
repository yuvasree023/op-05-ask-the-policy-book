# LANDSCAPE — what exists, what we tried, what we kept

One page on the ecosystem for "ask a document corpus, with citations, roles
and dates", and where this solution landed.

## What exists

- **Managed RAG platforms** (vector DB + hosted LLM bundles): fast to demo,
  but the grading contract needs a verbatim-quote guarantee, role isolation
  provable by construction, an offline CPU reproduce path, and a token
  budget below a fixed reference. A black-box chain gives none of those
  controls.
- **Retrieval stacks**: lexical (BM25/TF-IDF), dense embeddings, hybrids,
  rerankers. The published reference is TF-IDF top-6 into a budget model.
  Rerankers are allowed but add a model dependency and tokens for no
  measured gain at 391 passages.
- **Generators**: budget-class hosted models (mini/nano/Flash/Haiku tier)
  or open-weight <= ~35B (MODELS.md). Free tiers exist for some (Gemini API,
  Groq); FREE_CREDITS.md lists candidates without guaranteeing current
  access — each must be verified against live pricing/limits before use.
- **Citation verification**: quote-as-substring checking against the source
  bytes (what the grader does), vs "snap to nearest sentence" repair.
  Snapping can detach a quote from the claim it supports; rejected.
- **Role isolation designs**: filter at retrieval (above-role text never
  enters any prompt) vs filter at generation (prompt includes everything,
  trust the model). Only the first makes leaks impossible rather than
  unlikely.

## What we tried

| Approach | Result | Kept? |
|---|---|---|
| Vendored quantized MiniLM (ONNX) embeddings + persistent Chroma | 23 MB model, CPU-only, offline; metadata filters for role/date | kept |
| Hybrid BM25 + vector, max-normalized weighted sum (0.3/0.7) | dev recall .933@6 / .966@10; beat RRF and BM25-only | kept |
| Pure sentence-idf citation choice | gold-section hit rate 0.521 | dropped |
| Extractive sentence-window composer (stage 3 default) | verbatim 1.0 by construction, 0 API tokens; wrong-passage and missing-conclusion failures on inspection, gold-section choice 0.667 | kept as offline fallback |
| Budget-class LLM generator over role-filtered evidence (stage 4) | measured below in EXPERIMENT_LOG.md | kept when a verified free key exists |
| "Snap to nearest sentence" quote repair | can detach quote from claim; support-bar risk | rejected |

## What we dropped and why

- **Reranker / query rewriting / fine-tuning**: no measured need at this
  corpus size; each adds tokens, dependencies and review surface.
- **Generation-time role filtering**: leaks are then model-behavior
  dependent. Retrieval-time isolation keeps above-role text out of every
  prompt, including external APIs.
- **Generic folder ingestion**: would invent roles/dates the isolation bars
  depend on. The published corpus format is the only accepted input.

## Honest note on the LLM generator

The LLM path needs network access and a provider key at run time; the
shipped default remains the offline extractive composer. Measured trade-offs
(tokens vs the 1,005/743 reference, judged correctness/support on dev) are
in EXPERIMENT_LOG.md with dates and raw outputs in results/.
