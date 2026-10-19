"""Stage 3 service tests: HTTP contract on an in-process server."""
import hashlib
import json
import sys
import tempfile
import threading
import unittest
import urllib.request
import urllib.error
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from policybook import service  # noqa: E402
from policybook.answer import Answerer  # noqa: E402
from policybook.ingest import run as ingest_run  # noqa: E402
from test_answer import MODEL_DIR, write_corpus  # noqa: E402


def post(port, path, payload, raw=None):
    data = raw if raw is not None else json.dumps(payload).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


class ServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        root = Path(cls._td.name)
        write_corpus(root / "corpus")
        assert ingest_run(["--corpus-dir", str(root / "corpus"),
                           "--index-dir", str(root / "index"),
                           "--report", str(root / "report.json")]) == 0
        service._State.answerer = Answerer(root / "index", MODEL_DIR, root / "corpus")
        service._State.ready = True
        cls.srv = service.ThreadingHTTPServer(("127.0.0.1", 0), service.Handler)
        cls.port = cls.srv.server_address[1]
        cls._t = threading.Thread(target=cls.srv.serve_forever, daemon=True)
        cls._t.start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls._td.cleanup()

    def test_healthz(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/healthz") as r:
            self.assertEqual(r.status, 200)

    def test_ask_contract(self):
        code, body = post(self.port, "/ask", {
            "id": "x1", "role": "legal", "as_of": "2023-01-01",
            "question": "What fraction of the anchor investor portion is reserved?"})
        self.assertEqual(code, 200)
        self.assertEqual(body["id"], "x1")
        self.assertIn(body["status"], ("answered", "not_in_corpus", "not_permitted"))
        self.assertIsInstance(body["answer"], str)
        self.assertIsInstance(body["citations"], list)

    def test_ask_role_blocked(self):
        code, body = post(self.port, "/ask", {
            "id": "x2", "role": "public", "as_of": "2023-01-01",
            "question": "What is the privileged litigation hold procedure?"})
        self.assertEqual(code, 200)
        self.assertEqual(body["status"], "not_permitted")
        self.assertEqual(body["citations"], [])

    def test_bad_role_rejected(self):
        code, _ = post(self.port, "/ask", {"id": "x3", "role": "root",
                                           "question": "anything", "as_of": "2023-01-01"})
        self.assertEqual(code, 400)

    def test_bad_json_rejected(self):
        code, _ = post(self.port, "/ask", None, raw=b"{not json")
        self.assertEqual(code, 400)

    def test_missing_fields_rejected(self):
        code, _ = post(self.port, "/ask", {"id": "x4"})
        self.assertEqual(code, 400)

    def test_404(self):
        code, _ = post(self.port, "/admin", {"id": "x5"})
        self.assertEqual(code, 404)


if __name__ == "__main__":
    unittest.main()
