# Vendored grader (self-check only)

`grader.py` and `rubric.md` are byte-identical copies of the published
Season 1 scorer, vendored so the dev self-check runs offline against the
exact deterministic checks the reviewer recomputes.

- Source: github.com/Deployment-inc/Deployment.inc-Hiring-Problems,
  `references/OP-05/grader.py`, `references/OP-05/rubric.md`
- Verified against RELEASE_MANIFEST.json (release season-1-2026-09-07):
  - grader.py sha256 8c3fcf3068e9940c130e7c54200d7f82bdcd1120bef60599918ee0998a718b80
  - rubric.md sha256 c6645365a7abc6aa628e2e6a7010d35e9fd2588c2b85d0e13f9fa4fb012f6592
- The LLM judge (rubric prompts A/B/C) needs a budget-class key
  (`LLM_BASE_URL`/`LLM_API_KEY`/`JUDGE_MODEL`) and is optional; the default
  self-check runs `--no-judge` (deterministic checks only).
