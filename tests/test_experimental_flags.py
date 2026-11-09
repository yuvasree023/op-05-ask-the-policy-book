"""Experimental generator flags: OFF by default, ON only when explicitly set.

OP05_GEN_DATE_CLAUSE=1 adds the date-asking prompt clause (DECISIONS 31,
isolated after the v3 dev run - experimental, not default behavior).
OP05_GEN_INSUFFICIENT_RETRY=1 allows one bounded second attempt when the
model says INSUFFICIENT on the eval path. Both default to October (v2)
behavior. Stub gateway only - no network, no key.
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

from policybook.answer import Answerer  # noqa: E402
from policybook.ingest import run as ingest_run  # noqa: E402
from tests.test_answer import write_corpus, MODEL_DIR  # noqa: E402

QUESTION = "What fraction of the anchor investor portion is reserved for domestic mutual funds?"
VALID = {"answer": "One third.",
         "citations": [{"doc_id": "PUB_ANCHOR", "section": "PUB_ANCHOR#s00000",
                        "quote": "One third of the anchor investor portion is reserved"}]}
INSUFFICIENT = {"answer": "INSUFFICIENT", "citations": []}


class SeqStub(BaseHTTPRequestHandler):
    """Returns queued response objects in order; records request payloads."""
    queue = []
    payloads = []

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        SeqStub.payloads.append(json.loads(body))
        obj = SeqStub.queue.pop(0) if SeqStub.queue else VALID
        data = json.dumps({"choices": [{"message": {"content": json.dumps(obj)}}],
                           "usage": {"prompt_tokens": 10, "completion_tokens": 5}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


class FlagTests(unittest.TestCase):
    def setUp(self):
        SeqStub.queue, SeqStub.payloads = [], []
        self.srv = HTTPServer(("127.0.0.1", 0), SeqStub)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self._td = tempfile.TemporaryDirectory()
        root = Path(self._td.name)
        write_corpus(root / "corpus")
        assert ingest_run(["--corpus-dir", str(root / "corpus"),
                           "--index-dir", str(root / "index"),
                           "--report", str(root / "ingest.json")]) == 0
        self.root = root
        self._old = os.environ.copy()
        os.environ.update({
            "OP05_GEN_BACKEND": "openai", "OP05_GEN_MODEL": "stub-model",
            "LLM_BASE_URL": f"http://127.0.0.1:{self.srv.server_address[1]}",
            "LLM_API_KEY": "stub"})
        os.environ.pop("OP05_GEN_DATE_CLAUSE", None)
        os.environ.pop("OP05_GEN_INSUFFICIENT_RETRY", None)
        self.ans = Answerer(root / "index", MODEL_DIR, root / "corpus")

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._old)
        self.srv.shutdown()
        self._td.cleanup()

    def _answer(self):
        return self.ans.answer("q1", QUESTION, "legal", "9999-12-31")

    def test_date_clause_off_by_default(self):
        SeqStub.queue = [VALID]
        res = self._answer()
        self.assertEqual(res.status, "answered")
        sent = SeqStub.payloads[0]["messages"][0]["content"]
        self.assertNotIn("Date-asking", sent)

    def test_date_clause_on_when_flagged(self):
        os.environ["OP05_GEN_DATE_CLAUSE"] = "1"
        SeqStub.queue = [VALID]
        self._answer()
        sent = SeqStub.payloads[0]["messages"][0]["content"]
        self.assertIn("Date-asking", sent)

    def test_insufficient_no_retry_by_default(self):
        SeqStub.queue = [INSUFFICIENT, VALID]
        res = self._answer()
        self.assertFalse(res.generator.startswith("llm:"))
        self.assertEqual(len(SeqStub.payloads), 1)  # no second attempt

    def test_insufficient_retry_when_flagged(self):
        os.environ["OP05_GEN_INSUFFICIENT_RETRY"] = "1"
        SeqStub.queue = [INSUFFICIENT, VALID]
        res = self._answer()
        self.assertEqual(res.status, "answered")
        self.assertTrue(res.generator.startswith("llm:"))
        self.assertEqual(len(SeqStub.payloads), 2)  # exactly one retry
        self.assertTrue(res.debug["gen_usage"].get("insufficient_retry_used"))

    def test_insufficient_retry_gives_up_after_one(self):
        os.environ["OP05_GEN_INSUFFICIENT_RETRY"] = "1"
        SeqStub.queue = [INSUFFICIENT, INSUFFICIENT, VALID]
        res = self._answer()
        self.assertFalse(res.generator.startswith("llm:"))
        self.assertEqual(len(SeqStub.payloads), 2)  # bounded: no third call


if __name__ == "__main__":
    unittest.main()
