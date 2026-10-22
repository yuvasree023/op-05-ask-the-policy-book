"""Stage 3: the /ask HTTP service (stdlib only - no new dependencies, so the
installed footprint and offline guarantees from stages 1-2 are unchanged).

Contract (references/OP-05/freshness_contract.md):
- POST /ask with {id, question, role, as_of} -> {id, status, answer, citations}
- GET /healthz -> HTTP 200 once the index is loaded and the service is ready
- the listen port comes from the PORT environment variable (default 8000)
- role is supplied by the trusted runner; any role string outside the
  published set is rejected with 400, and role text inside the question is
  never authoritative

The server is multi-threaded (the reference latency bar is measured at
concurrency 12); retrieval reads only in-memory structures built at startup.
Error responses are generic and carry no corpus content, question text or
internals - a stack trace can leak restricted text as surely as an answer.

Run:  CORPUS_DIR=./data/corpus PORT=8000 ./.venv/bin/python -m policybook.service
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .answer import Answerer
from .corpus import ROLES
from .llm import QuotaExceeded, TransientUnavailable

REPO_ROOT = Path(__file__).resolve().parent.parent
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
MAX_BODY = 64 * 1024


class _State:
    answerer = None
    ready = False


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # -- helpers -------------------------------------------------------------
    def _send(self, code, obj, headers=None):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # quiet default access log
        pass

    # -- routes ----------------------------------------------------------------
    def do_GET(self):
        if self.path == "/healthz":
            self._send(200 if _State.ready else 503,
                       {"status": "ok" if _State.ready else "starting"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/ask":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY:
            self._send(400, {"error": "bad request body"})
            return
        try:
            req = json.loads(self.rfile.read(length))
        except (ValueError, UnicodeDecodeError):
            self._send(400, {"error": "body must be a JSON object"})
            return
        if not isinstance(req, dict):
            self._send(400, {"error": "body must be a JSON object"})
            return
        qid, question, role = req.get("id"), req.get("question"), req.get("role")
        as_of = req.get("as_of") or "9999-12-31"
        if (not isinstance(qid, str) or not qid
                or not isinstance(question, str) or not question.strip()
                or role not in ROLES or not isinstance(as_of, str)
                or not DATE.match(as_of)):
            self._send(400, {"error": "need {id: str, question: str, "
                             "role: public|ops|legal, as_of: YYYY-MM-DD}"})
            return
        if not _State.ready:
            self._send(503, {"error": "index not ready"})
            return
        t0 = time.perf_counter()
        try:
            res = _State.answerer.answer(qid, question, role, as_of)
        except (TransientUnavailable, QuotaExceeded):
            # Model path temporary failure (timeout/stall/provider 503/quota):
            # generic temporary error + Retry-After, never a silent
            # extractive fallback. No internals or corpus text leak.
            print(f"/ask {qid}: generator temporarily unavailable", file=sys.stderr)
            self._send(503, {"error": "generator temporarily unavailable"},
                       headers={"Retry-After": "5"})
            return
        except Exception:  # noqa: BLE001 - never leak internals or corpus text
            print(f"/ask {qid}: unhandled error", file=sys.stderr)
            self._send(500, {"error": "internal error"})
            return
        # Every served answer names its generator, so an extractive
        # offline-fallback answer is never indistinguishable from a model one.
        self._send(200, res.public_dict(), headers={"X-Generator": res.generator})
        print(f"/ask {qid}: {res.status} {res.latency_ms:.0f}ms "
              f"generator={res.generator} "
              f"(total {(time.perf_counter() - t0) * 1000:.0f}ms)", file=sys.stderr)


def main(argv=None) -> int:
    corpus_dir = os.environ.get("CORPUS_DIR", str(REPO_ROOT / "data" / "corpus"))
    index_dir = os.environ.get("INDEX_DIR", str(REPO_ROOT / ".index"))
    model_dir = os.environ.get("MODEL_DIR", str(REPO_ROOT / "models" / "all-MiniLM-L6-v2"))
    port = int(os.environ.get("PORT", "8000"))
    print(f"loading index {index_dir} and corpus {corpus_dir} ...", file=sys.stderr)
    _State.answerer = Answerer(index_dir, model_dir, corpus_dir, live=True)
    a = _State.answerer
    if a.gen_backend == "openai":
        mode = ("auto-activated on reviewer gateway credentials"
                if a.gen_auto else "explicitly enabled")
        print(f"generator: llm:{a.gen_model or 'gateway-default'} ({mode})",
              file=sys.stderr)
    else:
        print("generator: extractive (offline fallback - no LLM_BASE_URL/"
              "LLM_API_KEY credentials; answers are labeled extractive)",
              file=sys.stderr)
    _State.ready = True
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    srv.daemon_threads = True
    print(f"OP-05 /ask service ready on port {port}", file=sys.stderr)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
