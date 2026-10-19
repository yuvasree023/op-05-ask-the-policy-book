"""Row-level checkpointing: a killed run must leave partial evidence."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from policybook import answer_eval  # noqa: E402
from tests.test_answer import write_corpus, MODEL_DIR  # noqa: E402
from policybook.ingest import run as ingest_run  # noqa: E402


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_written_per_row(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            corpus = td / "corpus"
            write_corpus(corpus)
            index = td / "index"
            ingest_run(["--corpus-dir", str(corpus), "--index-dir", str(index), "--report", str(td / "ingest_report.json")])
            qs = td / "qs.jsonl"
            rows = [
                {"id": "q1", "question": "What fraction is reserved for mutual funds?",
                 "role": "public", "expected_status": "answered"},
                {"id": "q2", "question": "Who is the CEO of Mars?",
                 "role": "public", "expected_status": "refused_out_of_corpus"},
            ]
            qs.write_text("".join(json.dumps(r) + "\n" for r in rows))
            out = td / "answers.jsonl"
            env = {"CORPUS_DIR": str(corpus), "INDEX_DIR": str(index),
                   "MODEL_DIR": str(MODEL_DIR), "OP05_GEN_BACKEND": "extractive"}
            old = os.environ.copy()
            os.environ.update(env)
            try:
                rc = answer_eval.main(["--questions", str(qs), "--out", str(out),
                                       "--report", str(td / "report.json")])
            finally:
                os.environ.clear()
                os.environ.update(old)
            self.assertIn(rc, (0, 1))  # 1 = metric gates failed, checkpointing still must hold
            checkpoint = out.with_suffix(".checkpoint.jsonl")
            self.assertTrue(checkpoint.exists())
            ck_rows = [json.loads(l) for l in checkpoint.read_text().splitlines()]
            self.assertEqual(len(ck_rows), 2)           # all rows checkpointed
            self.assertTrue(out.exists())
            self.assertEqual([r["id"] for r in ck_rows], ["q1", "q2"])


if __name__ == "__main__":
    unittest.main()
