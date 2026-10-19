# MEMO — to the Head of Compliance

**What it does.** Ask the policy book a question in plain language and get a
short answer with the exact passage it came from - document, section and a
verbatim quote you can check in seconds. It knows three access levels
(public, ops, legal) and answers only from what the requester may see. It
knows dates: ask "as of" a day and it answers from the rules in force that
day, naming when a later version changes the answer.

**What it refuses.** Two kinds of "no", kept distinct on purpose:
- *Not in the book*: the corpus does not contain the answer. It says so and
  guesses nothing.
- *Not permitted*: an answer exists but sits above the requester's access
  level. It says that something exists and access is denied - without naming
  or quoting the restricted document. The other option (pretending no rule
  exists) is the dangerous one: a compliance officer told there is no rule
  will act as though there is none.

**What a wrong answer costs.** A confidently wrong citation sends a branch
to a withdrawn circular - the exact failure that motivated this system. The
defences, in order: answers come only from retrieved evidence; every quote
is machine-verified against the source bytes before it is shown; anything
the verifier cannot confirm is dropped or refused rather than smoothed over.
Measured dev-set numbers and their gaps are in EXPERIMENT_LOG.md; the
held-out grading is run by the reviewers, not claimed by us.

**How you would know it went stale.** The index is keyed to the corpus
bytes. When a regulator amends something, replace the corpus folder and
restart `scripts/reproduce.sh`; the system detects the change by content
hash and re-indexes. A document superseded as of the question's date is
never cited as current. If answers start citing withdrawn material, the
corpus input - not the model - is the first place to look.

**What a year of upkeep takes.** Swapping in amendment packs as regulators
publish them (the restart path above), re-running the dev self-check after
each swap, and re-checking provider pricing/limits if the hosted generator
is used. The offline default (no LLM) keeps working with zero external
dependencies; the LLM generator is an optional quality upgrade, not a
requirement.

**Residual risk, honestly.** The refusal thresholds and retrieval weights
are tuned on 146 development questions; thin margins on out-of-corpus
refusal are documented in EXPERIMENT_LOG.md. The LLM-judged correctness and
support bars are measured on dev with a substitute judge, not the pinned
reviewer judge - treat those numbers as indicative, not final. If the system
is ever asked to answer beyond the shipped corpus format (scanned PDFs,
emails, new document types), pause rollout and re-validate first: ingestion
refuses unknown formats loudly rather than guessing, and that is the
condition under which questions should go to a person.
