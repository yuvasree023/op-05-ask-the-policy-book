# v5 dev run state (2026-11-23)

Variant: evidence pool k=12 (signal_k pinned 10), r.hits[:12], char budget
3000, rewritten generation prompt (exact-figure/derived-results, temporal,
contradiction, single-contiguous-quote rules + DOCUMENT TIMELINE block).
Commit 93d701a.

Generation: COMPLETE. 146/146 rows, 100 llm-answered / 17 extractive-
answered / 8 not_in_corpus / 21 not_permitted. 164,578 input + 28,752
output tokens; 0.1213 USD at list price, 0 actual (free tier). Per-question
input average is OVER the 1005-token reference bar (~1127 across 146
questions): a cost-constrained re-run with OP05_GEN_EV_CHAR_BUDGET=2600 is
the planned check if the judged metrics justify keeping the variant.

Judging: IN PROGRESS at snapshot time (substitute judge
gemini-3.5-flash-lite, JUDGE_DELAY_SEC=4.5). The verdict sidecar
results/raw/judge_verdicts_v5_2026-11-23.jsonl is content-keyed and
resumable: re-running the same grader command reuses cached verdicts.

Resume after a sandbox reset:
1. Clone repo, `python3 -m venv .venv`, pip install -r requirements.txt,
   `./.venv/bin/python -m policybook.ingest`.
2. Vault: write-file 'Gemini API key (OP-05)' password and
   'Gemini API key (OP-05 fallback, project op-05-eval-2)' password.
3. Run judge only:
   LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai \
   LLM_API_KEY=<primary> LLM_API_KEY_FALLBACK=<fallback> \
   JUDGE_MODEL=gemini-3.5-flash-lite JUDGE_DELAY_SEC=4.5 \
   JUDGE_VERDICTS_PATH=results/raw/judge_verdicts_v5_2026-11-23.jsonl \
   ./.venv/bin/python tools/op05_grader/grader.py \
     --pred results/raw/answers_dev_llm_v5_2026-11-23.jsonl \
     --gold data/questions_dev.jsonl \
     --corpus data/corpus/manifest.jsonl \
     --report results/raw/grader_dev_llm_v5_2026-11-23.json
4. Compare vs v2 (0.8403/0.7037) and v4A (0.8151/0.6475); keep only if
   both bars improve; then cost-bar variant OP05_GEN_EV_CHAR_BUDGET=2600.

drive_v5.sh is the original run script with the vault paths scrubbed; the
keys come from the vault entries named above, never from the repo.
