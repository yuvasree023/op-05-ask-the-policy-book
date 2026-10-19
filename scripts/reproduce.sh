#!/usr/bin/env bash
# OP-05 "Ask the Policy Book" - reproduce script.
#
#   ./scripts/reproduce.sh                      # ingest corpus, then serve POST /ask on $PORT
#   ./scripts/reproduce.sh --selfcheck          # ingest, then offline dev evaluation, then exit
#   CORPUS_DIR=/path/to/corpus ./scripts/reproduce.sh [--force] [--selfcheck]
#
# Review contract (references/OP-05/freshness_contract.md): the default
# invocation brings the service up - ingestion (rebuilding the index when the
# corpus changed, skipping when it did not) followed by the HTTP service:
#   POST /ask     {id, question, role, as_of} -> {id, status, answer, citations}
#   GET /healthz  200 once the index is loaded
# PORT defaults to 8000. To evaluate instead of serving, use --selfcheck
# (SUBMISSION_SCHEMA.md: subcommands, not a server that blocks evaluation):
#   --selfcheck runs the retrieval dev eval, answers the dev questions
#   (results/answers_dev.jsonl) and grades them with the vendored published
#   grader (tools/op05_grader, deterministic checks; add LLM_BASE_URL/
#   LLM_API_KEY/JUDGE_MODEL + JUDGE=1 for the rubric judge).
#
# A swapped or modified corpus is detected by content hash and re-indexed;
# an unchanged corpus skips re-embedding (freshness: stop, swap CORPUS_DIR,
# restart with the same script - a stale index never survives a restart).
#
# Network: model, corpus, dev questions and the vendored grader ship in the
# repo, so ingestion, retrieval, answering and the deterministic self-check
# run fully offline. The ONE network touchpoint is pip installing
# requirements into .venv on first run; vendor/wheels/ (scripts/vendor_wheels.sh)
# removes even that. --no-cache-dir keeps pip from writing a wheel cache.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
if [ ! -d .venv ]; then
  "$PYTHON" -m venv .venv
fi
if [ -d vendor/wheels ]; then
  ./.venv/bin/pip install -q --no-cache-dir --disable-pip-version-check \
    --no-index --find-links vendor/wheels -r requirements.txt
else
  ./.venv/bin/pip install -q --no-cache-dir --disable-pip-version-check \
    -r requirements.txt
fi

export CORPUS_DIR="${CORPUS_DIR:-./data/corpus}"
export INDEX_DIR="${INDEX_DIR:-./.index}"

SELFCHECK=0
ARGS=()
for a in "$@"; do
  case "$a" in
    --selfcheck) SELFCHECK=1 ;;
    *) ARGS+=("$a") ;;
  esac
done

./.venv/bin/python -m policybook.ingest "${ARGS[@]+"${ARGS[@]}"}"

if [ "$SELFCHECK" = "1" ]; then
  QUESTIONS_FILE="${QUESTIONS_FILE:-./data/questions_dev.jsonl}"
  if [ "${RETRIEVAL_EVAL:-1}" != "0" ] && [ -f "$QUESTIONS_FILE" ]; then
    ./.venv/bin/python -m policybook.retrieve_eval --questions "$QUESTIONS_FILE"
  fi
  JUDGE_ARGS=()
  if [ "${JUDGE:-0}" = "1" ]; then JUDGE_ARGS+=(--judge); fi
  ./.venv/bin/python -m policybook.answer_eval --questions "$QUESTIONS_FILE" "${JUDGE_ARGS[@]+"${JUDGE_ARGS[@]}"}"
  exit 0
fi

export PORT="${PORT:-8000}"
exec ./.venv/bin/python -m policybook.service
