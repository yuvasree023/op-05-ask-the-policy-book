#!/bin/sh
# Durable checkpoint loop: while a judged grading run writes verdicts to
# results/judge_verdicts.jsonl, push that sidecar to the judge-checkpoints
# branch every 3 minutes via a SEPARATE WORKTREE (never touches the running
# checkout), so a sandbox rebuild loses at most 3 minutes of grading.
# Sidecar content is hashes + verdict tokens only (no prompts, no secrets).
# History is RETAINED (plain sequential commits, no amend, no force-push)
# until final verified artifacts supersede the checkpoints.
set -e
cd "$(dirname "$0")/.."
export GIT_SSH_COMMAND='ssh -i /home/sandbox/.ssh/op05_deploy -o StrictHostKeyChecking=accept-new'
WT=/tmp/judge-ckpt-worktree
git worktree add "$WT" -B judge-checkpoints >/dev/null 2>&1 || git worktree add "$WT" judge-checkpoints >/dev/null 2>&1 || true
while true; do
  sleep 180
  if [ -f results/judge_verdicts.jsonl ]; then
    mkdir -p "$WT/results"
    cp results/judge_verdicts.jsonl "$WT/results/judge_verdicts.jsonl"
    (cd "$WT" && git add -f results/judge_verdicts.jsonl \
      && git commit -q -m "judge verdicts checkpoint $(date -u +%FT%TZ) ($(wc -l < results/judge_verdicts.jsonl) verdicts)" \
      && git push -q git@github.com:yuvasree023/op-05-ask-the-policy-book.git judge-checkpoints:judge-checkpoints) \
      && echo "[checkpoint] pushed $(wc -l < results/judge_verdicts.jsonl) verdicts at $(date +%T)"
  fi
done
