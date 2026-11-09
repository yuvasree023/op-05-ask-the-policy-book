"""Eval-runner regression: a run with model-generated rows must complete.

Covers the scoping defect where `key_fallback_rows` was initialized in
main() but missing from _iter_rows()'s nonlocal declaration, so ANY eval
run that produced a model-generated row crashed with UnboundLocalError.
Uses a local stub gateway - no network, no key.
"""
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from policybook import answer_eval  # noqa: E402
from policybook.ingest import run as ingest_run  # noqa: E402
from tests.test_answer import write_corpus, MODEL_DIR  # noqa: E402


class OkStub(BaseHTTPRequestHandler):
    """OpenAI-compatible stub: one verified answer per call."""

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        obj = {"answer": "One third.",
               "citations": [{"doc_id": "PUB_ANCHOR", "section": "PUB_ANCHOR#s00000",
                              "quote": "One third of the anchor investor portion is reserved"}]}
        body = json.dumps({"choices": [{"message": {"content": json.dumps(obj)}}],
                           "usage": {"prompt_tokens": 10, "completion_tokens": 5}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class EvalLLMRowTests(unittest.TestCase):
    def test_eval_run_with_llm_rows_completes(self):
        srv = HTTPServer(("127.0.0.1", 0), OkStub)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            with tempfile.TemporaryDirectory() as td:
                td = Path(td)
                corpus = td / "corpus"
                write_corpus(corpus)
                index = td / "index"
                self.assertEqual(ingest_run(["--corpus-dir", str(corpus),
                                             "--index-dir", str(index),
                                             "--report", str(td / "ingest.json")]), 0)
                qs = td / "qs.jsonl"
                qs.write_text(json.dumps({
                    "id": "q1",
                    "question": "What fraction of the anchor investor portion is reserved for domestic mutual funds?",
                    "role": "legal", "expected_status": "answered", "gold_answer": "One third of the anchor investor portion is reserved for domestic mutual funds."}) + "\n")
                out = td / "answers.jsonl"
                report = td / "report.json"
                env = {"CORPUS_DIR": str(corpus), "INDEX_DIR": str(index),
                       "MODEL_DIR": str(MODEL_DIR),
                       "OP05_GEN_BACKEND": "openai", "OP05_GEN_MODEL": "stub-model",
                       "OP05_GEN_DELAY_SEC": "0",
                       "LLM_BASE_URL": f"http://127.0.0.1:{srv.server_address[1]}",
                       "LLM_API_KEY": "stub"}
                old = os.environ.copy()
                os.environ.update(env)
                try:
                    rc = answer_eval.main(["--questions", str(qs), "--out", str(out),
                                           "--report", str(report)])
                finally:
                    os.environ.clear()
                    os.environ.update(old)
                self.assertEqual(rc, 0)
                rows = [json.loads(l) for l in out.read_text().splitlines()]
                self.assertEqual(len(rows), 1)
                self.assertTrue(rows[0]["generator"].startswith("llm:"))
                self.assertFalse(rows[0]["extractive_fallback"])
                summary = json.loads(report.with_suffix(".run.json").read_text())
                self.assertIn("key_fallback_rows", summary)
        finally:
            srv.shutdown()


if __name__ == "__main__":
    unittest.main()
