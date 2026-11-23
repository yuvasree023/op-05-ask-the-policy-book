"""Print the headline scores from the grader report (no network, read-only)."""
import json, sys
path = sys.argv[1] if len(sys.argv) > 1 else "results/grader_dev_report.json"
s = json.load(open(path))["summary"]
def f(k, label, bar=""):
    v = s.get(k)
    print(f"  {label:<34}{'n/a (needs judge: JUDGE=1)' if v is None else v}  {bar}")
print("=" * 60)
print("SCORE SUMMARY  (source: " + path + ")")
f("answer_accuracy", "answer accuracy", "(bar >= 0.88, judge only)")
f("citation_support_rate", "citation support", "(bar >= 0.82, judge only)")
f("citation_verbatim_rate", "citation verbatim", "(bar >= 0.92)")
f("answered_rate_on_answerable", "coverage", "(bar >= 0.87)")
f("ooc_refusal_rate", "out-of-corpus refusal", "(bar >= 0.95)")
f("not_permitted_rate", "role-blocked refusal", "(bar >= 0.87)")
f("role_violations", "role violations", "(must be 0)")
f("schema_valid_rate", "schema valid")
print("Accuracy/support need the rubric judge; without it they show n/a.")
print("=" * 60)
