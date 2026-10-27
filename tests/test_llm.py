"""Stage 3 optional-LLM-backend tests against a local stub endpoint.

No network, no key: the stub emulates an OpenAI-compatible chat-completions
route so the retry/verify plumbing is exercised without any live API call.
The backend stays OFF by default; these tests prove only the plumbing.
"""
import json
import os
import sys
import threading
import time
import unittest
from unittest import mock
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from policybook import llm  # noqa: E402


class Stub(BaseHTTPRequestHandler):
    calls = []

    def do_POST(self):
        Stub.calls.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        n = len(Stub.calls)
        if n == 1:
            obj = {"answer": "One third.", "citations": [
                {"doc_id": "D", "section": "D#s00000", "quote": "not in the passage"}]}
        else:
            obj = {"answer": "One third.", "citations": [
                {"doc_id": "D", "section": "D#s00000", "quote": "One third of the anchor"}]}
        body = json.dumps({"choices": [{"message": {"content": json.dumps(obj)}}],
                           "usage": {"prompt_tokens": 10, "completion_tokens": 5}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class Hit:
    def __init__(self):
        self.doc_id, self.section = "D", "D#s00000"
        self.text = "One third of the anchor investor portion is reserved."


class BusyStub(BaseHTTPRequestHandler):
    """Provider 503 'high demand': transient capacity, not quota."""
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        body = b'{"error": "high demand"}'
        self.send_response(503)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class LLMTests(unittest.TestCase):
    def test_retry_then_verified_citation(self):
        Stub.calls = []
        srv = HTTPServer(("127.0.0.1", 0), Stub)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            out = llm.generate("What fraction?", "2023-01-01", [Hit()],
                               verify_quote=lambda d, s, q: q in Hit().text,
                               model="stub-model",
                               base_url=f"http://127.0.0.1:{srv.server_address[1]}",
                               api_key="stub")
        finally:
            srv.shutdown()
        self.assertIsNotNone(out)
        answer, cits, usage = out
        self.assertEqual(len(Stub.calls), 2)          # bad quote -> one retry
        self.assertEqual(len(cits), 1)                # verified quote kept
        self.assertEqual(cits[0]["quote"], "One third of the anchor")
        self.assertEqual(usage["calls"], 2)
        self.assertEqual(usage["input_tokens"], 20)
        # request shape: no temperature field, max_completion_tokens only
        self.assertNotIn("temperature", Stub.calls[0])
        self.assertIn("max_completion_tokens", Stub.calls[0])

    def test_model_omitted_when_unnamed(self):
        # LLM_MODEL omittable: with no model named anywhere, the request
        # carries no "model" field and the gateway serves its default.
        Stub.calls = []
        srv = HTTPServer(("127.0.0.1", 0), Stub)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            with mock.patch.dict(os.environ, {}):
                os.environ.pop("OP05_GEN_MODEL", None)
                os.environ.pop("LLM_MODEL", None)
                out = llm.generate("What fraction?", "2023-01-01", [Hit()],
                                   verify_quote=lambda d, s, q: q in Hit().text,
                                   model="",
                                   base_url=f"http://127.0.0.1:{srv.server_address[1]}",
                                   api_key="stub")
        finally:
            srv.shutdown()
        self.assertIsNotNone(out)
        self.assertNotIn("model", Stub.calls[0])

    def test_one_attempt_live_path_raises_transient(self):
        # Approved 2026-10-22: on the live path a provider 503/timeout is one
        # bounded attempt, then a loud temporary error - never a silent
        # extractive fallback.
        srv = HTTPServer(("127.0.0.1", 0), BusyStub)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            with self.assertRaises(llm.TransientUnavailable):
                llm.generate("q", "2023-01-01", [Hit()],
                             verify_quote=lambda d, s, q: True, model="m",
                             base_url=f"http://127.0.0.1:{srv.server_address[1]}",
                             api_key="k", one_attempt=True)
        finally:
            srv.shutdown()

    def test_insufficient_returns_none(self):
        class Stub2(Stub):
            def do_POST(self):
                body = json.dumps({"choices": [{"message": {"content":
                    json.dumps({"answer": "INSUFFICIENT", "citations": []})}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        srv = HTTPServer(("127.0.0.1", 0), Stub2)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            out = llm.generate("Unknown?", "2023-01-01", [Hit()],
                               verify_quote=lambda d, s, q: True,
                               model="m", base_url=f"http://127.0.0.1:{srv.server_address[1]}",
                               api_key="k")
        finally:
            srv.shutdown()
        self.assertIsNone(out)


if __name__ == "__main__":
    unittest.main()


class QuotaStub(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        body = b'{"error": {"message": "rate limit reached"}}'
        self.send_response(429)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass



class KeyFallbackStub(BaseHTTPRequestHandler):
    """429s the primary key, serves the fallback key: key-level failover,
    same model and endpoint."""
    calls = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        auth = self.headers.get("Authorization", "")
        KeyFallbackStub.calls.append({"auth": auth, "model": body.get("model")})
        if auth != "Bearer k2":
            payload = b'{"error": {"message": "rate limit reached"}}'
            self.send_response(429)
        else:
            obj = {"answer": "One third.", "citations": [
                {"doc_id": "D", "section": "D#s00000", "quote": "One third of the anchor"}]}
            payload = json.dumps({"choices": [{"message": {"content": json.dumps(obj)}}],
                                  "usage": {"prompt_tokens": 10, "completion_tokens": 5}}).encode()
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):
        pass


class AdaptStub(BaseHTTPRequestHandler):
    calls = []

    def do_POST(self):
        AdaptStub.calls.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        if "max_completion_tokens" in AdaptStub.calls[-1]:
            body = b'{"error": {"message": "Unsupported parameter: max_completion_tokens"}}'
            self.send_response(400)
        else:
            obj = {"answer": "One third.", "citations": [
                {"doc_id": "D", "section": "D#s00000", "quote": "One third of the anchor"}]}
            body = json.dumps({"choices": [{"message": {"content": json.dumps(obj)}}],
                               "usage": {"prompt_tokens": 10, "completion_tokens": 5}}).encode()
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class Hit:
    def __init__(self):
        self.doc_id, self.section = "D", "D#s00000"
        self.text = "One third of the anchor investor portion is reserved."


class LLMTests2(unittest.TestCase):
    def test_quota_raises_loudly(self):
        srv = HTTPServer(("127.0.0.1", 0), QuotaStub)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            with self.assertRaises(llm.QuotaExceeded):
                llm.generate("q", "2023-01-01", [Hit()], verify_quote=lambda *a: True,
                             model="m", base_url=f"http://127.0.0.1:{srv.server_address[1]}",
                             api_key="k")
        finally:
            srv.shutdown()


    def test_key_fallback_converts_quota_429(self):
        KeyFallbackStub.calls = []
        srv = HTTPServer(("127.0.0.1", 0), KeyFallbackStub)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            with mock.patch.dict(os.environ, {"LLM_API_KEY_FALLBACK": "k2"}):
                out = llm.generate("What fraction?", "2023-01-01", [Hit()],
                                   verify_quote=lambda d, s, q: q in Hit().text,
                                   model="m", base_url=f"http://127.0.0.1:{srv.server_address[1]}",
                                   api_key="k1")
        finally:
            srv.shutdown()
        self.assertIsNotNone(out)
        self.assertTrue(out[2]["key_fallback_used"])
        self.assertEqual([c["auth"] for c in KeyFallbackStub.calls],
                         ["Bearer k1", "Bearer k2"])
        # STRICT: the failover changes the key only, never the model.
        self.assertEqual({c["model"] for c in KeyFallbackStub.calls}, {"m"})

    def test_key_fallback_exhaustion_raises_loudly(self):
        srv = HTTPServer(("127.0.0.1", 0), QuotaStub)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            with mock.patch.dict(os.environ, {"LLM_API_KEY_FALLBACK": "k2"}):
                with self.assertRaises(llm.QuotaExceeded):
                    llm.generate("q", "2023-01-01", [Hit()], verify_quote=lambda *a: True,
                                 model="m", base_url=f"http://127.0.0.1:{srv.server_address[1]}",
                                 api_key="k1")
        finally:
            srv.shutdown()

    def test_max_tokens_adaptation(self):
        AdaptStub.calls = []
        srv = HTTPServer(("127.0.0.1", 0), AdaptStub)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            out = llm.generate("What fraction?", "2023-01-01", [Hit()],
                               verify_quote=lambda d, s, q: q in Hit().text,
                               model="m", base_url=f"http://127.0.0.1:{srv.server_address[1]}",
                               api_key="k")
        finally:
            srv.shutdown()
        self.assertIsNotNone(out)
        self.assertEqual(len(AdaptStub.calls), 2)
        self.assertIn("max_tokens", AdaptStub.calls[1])
        self.assertNotIn("max_completion_tokens", AdaptStub.calls[1])
        self.assertEqual(out[2]["adaptations"], ["max_tokens"])

    def test_evidence_header_carries_temporal_meta(self):
        seen = {}

        class MetaStub(Stub):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                seen["prompt"] = body["messages"][0]["content"]
                obj = {"answer": "INSUFFICIENT", "citations": []}
                out = json.dumps({"choices": [{"message": {"content": json.dumps(obj)}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
        srv = HTTPServer(("127.0.0.1", 0), MetaStub)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            llm.generate("q", "2023-06-01", [Hit()], verify_quote=lambda *a: True,
                         model="m", base_url=f"http://127.0.0.1:{srv.server_address[1]}",
                         api_key="k",
                         meta={"D": {"title": "Master Circular", "effective_date": "2023-01-01",
                                     "later_version": "Master Circular 2",
                                     "later_date": "2024-01-01"}})
        finally:
            srv.shutdown()
        self.assertIn("in force 2023-01-01", seen["prompt"])
        self.assertIn("later version in force 2024-01-01", seen["prompt"])


class DeadEndpointTests(unittest.TestCase):
    def test_unreachable_endpoint_returns_none_not_crash(self):
        # nothing listening on this port: connection refused
        out = llm.generate("q", "2023-01-01", [Hit()], verify_quote=lambda *a: True,
                           model="m", base_url="http://127.0.0.1:9", api_key="k")
        self.assertIsNone(out)


class _TrickleResp:
    """Fake response whose body trickles in slower than the deadline."""
    def __init__(self, parts, delay=0.5):
        self._parts = list(parts)
        self._delay = delay
    def read1(self, n):
        if not self._parts:
            return b""
        time.sleep(self._delay)
        return self._parts.pop(0)


class ReadBoundedTests(unittest.TestCase):
    def test_fast_body_reads_fully(self):
        resp = _TrickleResp([b'{"ok":', b' true}'], delay=0)
        self.assertEqual(llm._read_bounded(resp, deadline=5), b'{"ok": true}')

    def test_slow_trickle_raises_transient_unavailable(self):
        # 3 parts, 0.5s apart each; 0.6s total deadline must trip mid-body.
        resp = _TrickleResp([b"a", b"b", b"c"], delay=0.5)
        t0 = time.monotonic()
        with self.assertRaises(llm.TransientUnavailable):
            llm._read_bounded(resp, deadline=0.6)
        self.assertLess(time.monotonic() - t0, 2.0)  # bounded, not hung
