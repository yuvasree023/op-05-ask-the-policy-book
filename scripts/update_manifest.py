"""Append judged dev runs v5..v9 to results/manifest.json (idempotent; read-only on raw files)."""
import hashlib, json, pathlib
R = pathlib.Path(__file__).resolve().parent.parent
raw = R / "results" / "raw"
man = json.load(open(R / "results" / "manifest.json"))
have = {r["run_id"] for r in man["runs"]}
RUNS = [  # (run_id, answers file, report, verdicts, generator label, status, date)
 ("dev-llm-v5-2026-11-23", "answers_dev_llm_v5_2026-11-23.jsonl", "grader_dev_llm_v5_2026-11-23_loweffort_2026-11-23.report.json", "judge_verdicts_v5_2026-11-23_loweffort_2026-11-23.jsonl", "gemini-3.5-flash-lite", "completed", "2026-11-23"),
 ("dev-llm-v5b2600-2026-11-23", "answers_dev_llm_v5b2600_2026-11-23.jsonl", "grader_dev_llm_v5b2600_loweffort_2026-11-23.report.json", "judge_verdicts_v5b2600_loweffort_2026-11-23.jsonl", "gemini-3.5-flash-lite", "completed", "2026-11-23"),
 ("dev-llm-v6-2026-11-23", "answers_dev_llm_v6_2026-11-23.jsonl", "grader_dev_llm_v6_loweffort_2026-11-23.report.json", "judge_verdicts_v6_loweffort_2026-11-23.jsonl", "gemini-3.5-flash-lite", "completed", "2026-11-23"),
 ("dev-llm-v7groq-2026-11-24", "answers_dev_llm_v7groq_2026-11-24.jsonl", "grader_dev_llm_v7groq_loweffort_2026-11-24.report.json", "judge_verdicts_v7groq_loweffort_2026-11-24.jsonl", "groq openai/gpt-oss-120b (~120B parameters: OUTSIDE the budget class; diagnostic only, not a submittable generator)", "completed", "2026-11-24"),
 ("dev-llm-v7nim-2026-11-24", "answers_dev_llm_v7nim_2026-11-24.jsonl", "grader_dev_llm_v7nim_loweffort_2026-11-24.report.json", "judge_verdicts_v7nim_loweffort_2026-11-24.jsonl", "nvidia/nemotron-3.5-lightning-30b-a3b (thinking off)", "completed", "2026-11-24"),
 ("dev-llm-v8gemini-2026-11-24", "answers_dev_llm_v8gemini_2026-11-24.jsonl", "grader_dev_llm_v8gemini_loweffort_2026-11-24.report.json", "judge_verdicts_v8gemini_loweffort_2026-11-24.jsonl", "gemini-3.5-flash-lite (prompt rule + broad retry; reverted)", "completed; reverted", "2026-11-24"),
 ("dev-llm-v9nim-2026-11-24", "answers_dev_llm_v9nim_2026-11-24.jsonl", "grader_dev_llm_v9nim_loweffort_2026-11-24.report.json", "judge_verdicts_v9nim_loweffort_2026-11-24.jsonl", "nvidia/nemotron-3.5-lightning-30b-a3b (thinking off), quote-expand + broad retry + numeric effort", "completed", "2026-11-24"),
 ("dev-llm-v9groq-2026-11-25", "answers_dev_llm_v9groq_2026-11-25.jsonl", "grader_dev_llm_v9groq_loweffort_2026-11-25.report.json", "judge_verdicts_v9groq_loweffort_2026-11-25.jsonl", "groq openai/gpt-oss-120b (~120B parameters: OUTSIDE the budget class; diagnostic only, not a submittable generator), quote-expand + broad retry + numeric effort", "completed", "2026-11-25"),
]
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
added = 0
for rid, ans, rep, ver, gen, status, date in RUNS:
    if rid in have or not (raw / rep).exists():
        continue
    entry = {"run_id": rid, "model": f"{gen}; gemini-3.5-flash-lite substitute judge (diagnostic only, not the official gpt-4.1-mini-2025-04-14 judge)",
             "status": status, "timestamp": date,
             "metrics": json.load(open(raw / rep))["summary"], "raw": {}}
    nj = raw / ans.replace(".jsonl", "").replace("answers_dev_llm_", "grader_dev_llm_")
    runj = list(raw.glob(f"grader_dev_llm_{ans.split('answers_dev_llm_')[1].replace('.jsonl','')}.nojudge.run.json"))
    if runj:
        t = json.load(open(runj[0])).get("tokens")
        entry["tokens"] = {"generation_input": t.get("input"), "generation_output": t.get("output")} if t else None
    for f in (ans, rep, ver):
        if (raw / f).exists():
            entry["raw"][f] = {"path": f"results/raw/{f}", "sha256": sha(raw / f)}
    man["runs"].append(entry); added += 1
man["note"] = man.get("note", "") + " Runs v5-v9 added 2026-11-25; their generators (Groq gpt-oss-120b) are outside the budget class where noted and are diagnostic only."
json.dump(man, open(R / "results" / "manifest.json", "w"), indent=2)
print("added", added)
