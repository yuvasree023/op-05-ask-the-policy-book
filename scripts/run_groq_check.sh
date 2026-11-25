#!/bin/bash
# One Groq generation run + one judged pass, single stream. Usage:
#   scripts/run_groq_check.sh base   # flags default-on (the v9groq setup)
#   scripts/run_groq_check.sh trim   # same plus OP05_TRIM_INPUT=1
# Needs .env with GROQ_API_KEY (and GEMINI_API_KEY for the judge). Keys are never printed.
set -u
cd "$(dirname "$0")/.."
MODE=${1:-base}
set -a; . ./.env; set +a
: "${GROQ_API_KEY:?put GROQ_API_KEY in .env}"
TAG=$MODE-$(date +%Y%m%d-%H%M)
OUT=results/raw/answers_dev_llm_groq_$TAG.jsonl
./.venv/bin/python -m policybook.ingest > /dev/null
EXTRA=""; [ "$MODE" = "trim" ] && EXTRA="OP05_TRIM_INPUT=1"
env $EXTRA LLM_BASE_URL=https://api.groq.com/openai/v1 LLM_API_KEY="$GROQ_API_KEY" OP05_GEN_BACKEND=openai OP05_GEN_STRICT=1 \
  OP05_GEN_MODEL=openai/gpt-oss-120b OP05_GEN_DELAY_SEC=14 OP05_GEN_EV_CHAR_BUDGET=${OP05_GEN_EV_CHAR_BUDGET:-2200} \
  OP05_GEN_INSUFFICIENT_RETRY=1 OP05_GEN_RETRY_EV_CHAR_BUDGET=3500 OP05_EVAL_RESUME=1 \
  OP05_PRICE_IN_PER_M=0.15 OP05_PRICE_OUT_PER_M=0.60 \
  ./.venv/bin/python -u -m policybook.answer_eval --questions data/questions_dev.jsonl --out "$OUT" \
  --report results/raw/grader_groq_$TAG.nojudge.json || { echo "generation stopped (rerun the same command to resume; if it says quota, wait for the daily reset)"; exit 1; }
if [ -n "${GEMINI_API_KEY:-}" ]; then
  JUDGE_VERDICTS_PATH=results/raw/judge_verdicts_groq_$TAG.jsonl LLM_API_KEY="$GEMINI_API_KEY" LLM_API_KEY_FALLBACK="" \
  JUDGE_MODEL=gemini-3.5-flash-lite JUDGE_REASONING_EFFORT=low JUDGE_DELAY_SEC=3.5 JUDGE_HTTP_TIMEOUT=60 \
  ./.venv/bin/python -u tools/op05_grader/grader.py --pred "$OUT" --gold data/questions_dev.jsonl \
    --corpus data/corpus/manifest.jsonl --judge-base-url https://generativelanguage.googleapis.com/v1beta/openai \
    --judge-model gemini-3.5-flash-lite --report results/raw/grader_groq_$TAG.report.json > /dev/null \
    || echo "judge stopped; rerun the same judge step (it resumes) after fixing the cause"
  REPORT=results/raw/grader_groq_$TAG.report.json
else
  echo "No GEMINI_API_KEY: skipping judge, accuracy/support will be n/a"; REPORT=results/raw/grader_groq_$TAG.nojudge.json
fi
./.venv/bin/python - "$REPORT" "$OUT" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]))["summary"]; rows=[json.loads(l) for l in open(sys.argv[2])]
print("accuracy   ",s.get("answer_accuracy"),"(bar >= 0.88)")
print("support    ",s.get("citation_support_rate"),"(bar >= 0.82)")
print("verbatim   ",s.get("citation_verbatim_rate"),"(bar >= 0.92)")
print("coverage   ",s.get("answered_rate_on_answerable"),"(bar >= 0.87)")
print("judge_errors",s.get("judge_errors"),"(should be 0)")
PY
RUNJ=$(ls results/raw/grader_groq_$TAG.nojudge.run.json 2>/dev/null)
[ -n "$RUNJ" ] && ./.venv/bin/python -c "
import json,sys
r=json.load(open('$RUNJ')); t=r['tokens']; n=r['questions']
print('input tokens/question ', round(t['input']/n), '(bar <= 1005)')
print('output tokens/question', round(t['output']/n), '(bar <= 743)')"
