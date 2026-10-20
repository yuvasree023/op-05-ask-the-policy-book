#!/bin/sh
# Durable checkpoint loop: while a judged grading run writes verdicts to
# results/judge_verdicts.jsonl, push that sidecar to the judge-checkpoints
# branch every 3 minutes so a sandbox rebuild loses at most 3 minutes of
# grading. Sidecar content is hashes + verdict tokens only (no prompts, no
# secrets) - safe for the private repo. Force-updated branch, squashed away
# after the final report lands.
set -e
cd "$(dirname "$0")/.."
export GIT_SSH_COMMAND='ssh -i /home/sandbox/.ssh/op05_deploy -o StrictHostKeyChecking=accept-new'
while true; do
  sleep 180
  if [ -f results/judge_verdicts.jsonl ]; then
    git checkout -q -B judge-checkpoints
    git add -f results/judge_verdicts.jsonl
    git commit -q -m "judge verdicts checkpoint $(date -u +%FT%TZ)" --amend 2>/dev/null \
      || git commit -q -m "judge verdicts checkpoint $(date -u +%FT%TZ)"
    git push -q -f git@github.com:yuvasree023/op-05-ask-the-policy-book.git judge-checkpoints:judge-checkpoints \
      && echo "[checkpoint] pushed $(wc -l < results/judge_verdicts.jsonl) verdicts at $(date +%T)"
    git checkout -q main
  fi
done
