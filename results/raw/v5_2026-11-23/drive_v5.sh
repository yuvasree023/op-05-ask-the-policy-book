#!/bin/bash
cd /home/sandbox/op05
export LLM_BASE_URL='https://generativelanguage.googleapis.com/v1beta/openai'
export LLM_API_KEY="$(cat VAULT_KEY_FILE)"
export LLM_API_KEY_FALLBACK="$(cat VAULT_KEY_FILE)"
export OP05_GEN_BACKEND=openai OP05_GEN_MODEL=gemini-3.5-flash-lite OP05_GEN_STRICT=1
export OP05_GEN_DELAY_SEC=4.5
export OP05_PRICE_IN_PER_M=0.30 OP05_PRICE_OUT_PER_M=2.50
export JUDGE_MODEL=gemini-3.5-flash-lite JUDGE_DELAY_SEC=4.5
export JUDGE_VERDICTS_PATH=results/raw/judge_verdicts_v5_2026-11-23.jsonl
exec ./.venv/bin/python -m policybook.answer_eval \
  --questions data/questions_dev.jsonl \
  --out results/raw/answers_dev_llm_v5_2026-11-23.jsonl \
  --report results/raw/grader_dev_llm_v5_2026-11-23.json --judge
