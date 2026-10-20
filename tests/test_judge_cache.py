"""Substitute-judge verdict cache: crash resume, no duplicate calls,
errors distinct from verdicts, config-change invalidation."""
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
sys.path.insert(0, str(REPO / "tools" / "op05_grader"))

import grader  # noqa: E402
from tests.test_answer import write_corpus  # noqa: E402


class JudgeStub(BaseHTTPRequestHandler):
    calls = 0
    fail_after = None      # when set: return 429 once calls reach this

    def do_POST(self):
        JudgeStub.calls += 1
        raw = self.rfile.read(int(self.headers["Content-Length"]))
        n = JudgeStub.calls
        if JudgeStub.fail_after is not None and n >= JudgeStub.fail_after:
            code, body = 429, b'{"error": {"code": 429, "message": "quota"}}'
        else:
            prompt = json.loads(raw)["messages"][1]["content"]
            if "PASSAGE (from" in prompt:
                token = "SUPPORTS"
            elif "GOLD:" in prompt:
                token = "CORRECT"
            else:
                token = "HEDGED"
            code, body = 200, json.dumps(
                {"choices": [{"message": {"content": token}}], "usage": {}}).encode()
        self.send_response(code)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        root = Path(self.td.name)
        write_corpus(root / "corpus")
        # gold: 4 answered + 1 ooc + 1 not_permitted
        gold = []
        pred = []
        # distinct question/gold/answer per row so every judge prompt is unique
        for i in range(4):
            gold.append({"id": f"q{i}", "question": f"What fraction is reserved in case {i}?",
                         "role": "public", "expected_status": "answered",
                         "gold_answer": f"One third in case {i}.", "tags": []})
            pred.append({"id": f"q{i}", "status": "answered",
                         "answer": f"One third of the anchor investor portion in case {i}.",
                         "citations": [{"doc_id": "PUB_ANCHOR", "section": "PUB_ANCHOR#s00000",
                                        "quote": "One third of the anchor investor portion"}],
                         "confidence": 0.9, "cost_usd": 0.0, "latency_ms": 1.0})
        gold.append({"id": "ooc", "question": "Who is the CEO of Mars?", "role": "public",
                     "expected_status": "not_in_corpus", "tags": []})
        pred.append({"id": "ooc", "status": "not_in_corpus", "answer": "", "citations": [],
                     "confidence": 0.1, "cost_usd": 0.0, "latency_ms": 1.0})
        gold.append({"id": "np", "question": "Privileged hold?", "role": "public",
                     "expected_status": "not_permitted", "tags": []})
        pred.append({"id": "np", "status": "not_permitted", "answer": "", "citations": [],
                     "confidence": 0.1, "cost_usd": 0.0, "latency_ms": 1.0})
        (root / "gold.jsonl").write_text("".join(json.dumps(r) + "\n" for r in gold))
        (root / "pred.jsonl").write_text("".join(json.dumps(r) + "\n" for r in pred))
        self.root = root

    def tearDown(self):
        self.td.cleanup()

    def _stub(self, fail_after=None):
        JudgeStub.calls = 0
        JudgeStub.fail_after = fail_after
        srv = HTTPServer(("127.0.0.1", 0), JudgeStub)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv

    def _run(self, srv, verdicts_path, report_name="report.json"):
        env = os.environ.copy()
        env["LLM_API_KEY"] = "stub"
        env["JUDGE_VERDICTS_PATH"] = str(verdicts_path)
        old = os.environ.copy()
        os.environ.update(env)
        try:
            argv = sys.argv
            sys.argv = ["grader.py", "--pred", str(self.root / "pred.jsonl"),
                        "--gold", str(self.root / "gold.jsonl"),
                        "--corpus", str(self.root / "corpus" / "manifest.jsonl"),
                        "--judge-base-url", f"http://127.0.0.1:{srv.server_address[1]}",
                        "--judge-model", "stub-judge",
                        "--report", str(self.root / report_name)]
            try:
                grader.main()
            finally:
                sys.argv = argv
        finally:
            os.environ.clear()
            os.environ.update(old)
        return json.loads((self.root / report_name).read_text())

    def test_crash_resume_no_duplicate_calls(self):
        vp = self.root / "verdicts.jsonl"
        # run 1: crash (429) after 3 judged calls -> SystemExit, sidecar kept
        srv1 = self._stub(fail_after=3)
        try:
            with self.assertRaises(SystemExit):
                self._run(srv1, vp)
        finally:
            srv1.shutdown()
        sidecar = [json.loads(l) for l in vp.read_text().splitlines()]
        self.assertEqual(len([e for e in sidecar if e["ok"]]), 2)  # 2 verdicts stored before 3rd call died
        # run 2: resume with a fresh stub - must only make the remaining calls
        srv2 = self._stub()
        try:
            rep = self._run(srv2, vp)
        finally:
            srv2.shutdown()
        self.assertEqual(JudgeStub.calls, 8 - 2)          # 8 total needed, 2 reused
        self.assertEqual(rep["summary"]["judge_verdicts_reused"], 2)
        self.assertEqual(rep["summary"]["judge_calls"], 6)
        # control: single uninterrupted run, identical verdict content
        srv3 = self._stub()
        try:
            ctrl = self._run(srv3, self.root / "verdicts_control.jsonl", "report_ctrl.json")
        finally:
            srv3.shutdown()
        self.assertEqual(JudgeStub.calls, 8)
        strip = lambda r: [{k: row[k] for k in ("id", "correct", "support_ok", "status_ok")}
                           for row in r["rows"]]
        self.assertEqual(strip(rep), strip(ctrl))

    def test_config_change_invalidates_cache(self):
        vp = self.root / "verdicts.jsonl"
        srv1 = self._stub()
        try:
            self._run(srv1, vp)
        finally:
            srv1.shutdown()
        self.assertEqual(JudgeStub.calls, 8)
        # change the pred input: cache must be ignored entirely
        rows = [json.loads(l) for l in (self.root / "pred.jsonl").read_text().splitlines()]
        rows[0]["answer"] = "About one third of the anchor investor portion in case 0."
        (self.root / "pred.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        srv2 = self._stub()
        try:
            rep = self._run(srv2, vp)
        finally:
            srv2.shutdown()
        self.assertEqual(JudgeStub.calls, 8)              # no reuse
        self.assertEqual(rep["summary"]["judge_verdicts_reused"], 0)


if __name__ == "__main__":
    unittest.main()
