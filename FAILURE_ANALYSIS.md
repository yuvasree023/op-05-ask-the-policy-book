# Failure analysis - OP-05 dev set

This is the current offline taxonomy for the v4A dev run (2026-11-10). It uses
published dev labels and stored run artifacts only. It does not tune held-out
rows, call a model, or replace the official judge.

## Current measured position

| metric | v4A result | bar |
|---|---:|---:|
| answer accuracy, substitute judge | 0.8151 | >= 0.88 |
| citation support, substitute judge | 0.6475 | >= 0.82 |
| coverage on answerable dev rows | 0.9832 | >= 0.87 |
| citation verbatim rate | 1.0 | >= 0.92 |
| out-of-corpus refusal | 1.0, 0 fabrications | >= 0.95, 0 |
| role-blocked refusal | 1.0 | >= 0.87 |
| deterministic role violations | 0 | 0 |

The best completed dev measurement remains the older v2 set at 0.8403 accuracy
and 0.7037 support. Neither v2 nor v4A clears both judged bars. The judge is
gemini-3.5-flash-lite, a diagnostic substitute, not the pinned reviewer judge.

## What failed in v4A

- 22 of 119 answerable rows were judged incorrect:
  16 extractive fallback rows and
  6 model rows.
- 49 citations were judged unsupported across
  33 answerable rows. 17 of those
  rows had a correct answer with at least one weak citation, so support is a
  separate failure mode from correctness.
- For the 22 incorrect rows, the exact model evidence pack
  contained all gold sections in 13 cases and missed gold evidence in 9.
  The misses split into 3 character-budget exclusions at retrieval ranks
  5/5/7, 5 candidates ranked outside the first eight (ranks 9/9/10/17/19),
  and 1 beyond rank 20.

| category | n | incorrect | rows with unsupported citation |
|---|---:|---:|---:|
| contradiction_detection | 8 | 0 | 6 |
| internal_policy | 2 | 0 | 0 |
| numerical_reasoning | 27 | 6 | 10 |
| regulatory_interpretation | 63 | 14 | 13 |
| temporal_reasoning | 19 | 2 | 4 |

## Failure classes and interventions

1. **Extractive fallback is not a quality-safe substitute.** It accounts for
   16/22 v4A errors. It often quotes a nearby rule instead of deriving the
   requested date, amount, threshold or comparison. The v4A retry experiment
   did not improve the aggregate and the flags remain OFF.
2. **Model reasoning and answer selection still lose with evidence present.**
   Six model rows were incorrect; 13/22 total incorrect rows had all gold
   sections in the model pack. This points at prompt/composition and answer
   selection, not only retrieval recall.
3. **Retrieval and context packing are distinct bottlenecks.** Equal RRF put
   gold in 101/119 answerable packs. Lexical weight changes did not improve
   packed-gold coverage (still 101/119), while lexical-only reached 102/119
   with 15 row swaps and no quality evidence. No runtime ranking change was
   justified.
4. **Citation support is the largest gap.** Contradiction answers were 8/8
   correct but 6/8 rows had an unsupported citation, matching the split-quote
   pattern found in CON rows. The offline contiguous-span variant is preserved
   as a hypothesis only; its paired judge test stopped incomplete on HTTP 503
   and is not aggregate evidence.
5. **Temporal behavior passes deterministic controls but remains a judged
   risk.** v4A temporal accuracy was 0.8947 (n=19). The date clause fixed its
   target class in a small test but the full v3 run regressed, so it stays
   isolated OFF.

## Run-to-run intervention history

| change | result | decision |
|---|---|---|
| RRF retrieval and npz/json store | deterministic gates passed; retrieval recall@6 0.8908; reproducibility improved | keep |
| v2 model prompt | best completed dev judged set: 0.8403 / 0.7037 | keep as the recorded best, below bars |
| v3 date-asking clause | fixed target rows but full run regressed to 0.8235 / 0.6569 | keep flag OFF |
| v4A INSUFFICIENT retry | recovered-answer set fell to 0.8151 / 0.6475; 18 initial HTTP 503s remain unclassified | keep flag OFF |
| offline citation spans | structural checks passed; paired judged test incomplete after 16 verdicts | do not activate without recovery signal and completion |

Raw row-level detail: `results/failure_analysis_dev_v4A_2026-11-10.json`.
Run history and caveats: `EXPERIMENT_LOG.md`.
