"""Measurement harness tests use a local HTTP fixture, never a model."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
from scripts.measure_service import measure


class Stub(BaseHTTPRequestHandler):
    fail_first = False
    payloads = []

    def do_POST(self):
        obj = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        Stub.payloads.append(obj)
        if Stub.fail_first:
            self.send_response(503)
            self.send_header('Retry-After', '5')
            self.end_headers()
            self.wfile.write(b'{"error":"generic"}')
            return
        self.send_response(200)
        self.send_header('X-Generator', 'fixture')
        self.end_headers()
        self.wfile.write(json.dumps({'id':obj['id'], 'status':'answered', 'answer':'fixture', 'citations':[]}).encode())

    def log_message(self, *args):
        pass


class MeasurementTests(unittest.TestCase):
    def setUp(self):
        Stub.fail_first = False
        Stub.payloads = []
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Stub)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.temp = tempfile.TemporaryDirectory()
        self.out = Path(self.temp.name)
        self.url = 'http://127.0.0.1:' + str(self.server.server_port)
        self.questions = [{'id': str(i), 'question':'test', 'role':'public',
                           'as_of':'2020-01-01', 'gold_answer':'must not be sent'} for i in range(12)]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.temp.cleanup()

    def test_header_provenance_and_gold_exclusion(self):
        summary = measure(self.questions, self.url, self.out, concurrency=12, pace_seconds=0)
        self.assertEqual(summary['successful'], 12)
        self.assertTrue(all('gold_answer' not in q for q in Stub.payloads))
        attempts = [json.loads(l) for l in (self.out/'attempts.jsonl').read_text().splitlines()]
        self.assertTrue(all(r['generator']=='fixture' for r in attempts))
        self.assertTrue(all(r['elapsed_ms'] >= 0 for r in attempts))

    def test_503_stops_without_recovery_or_cause_guess(self):
        Stub.fail_first = True
        summary = measure(self.questions, self.url, self.out, pace_seconds=0)
        self.assertTrue(summary['stopped_early'])
        self.assertEqual(summary['attempted'], 1)
        self.assertEqual(summary['successful'], 0)
        row = json.loads((self.out/'attempts.jsonl').read_text())
        self.assertEqual(row['http_status'], 503)
        self.assertEqual(row['retry_after'], '5')
        self.assertIn('elapsed_ms', row)
        self.assertEqual(row['error_category'], 'http_error')
        self.assertEqual((self.out/'answers.jsonl').read_text(), '')

    def test_prior_evidence_not_overwritten(self):
        measure(self.questions[:1], self.url, self.out, pace_seconds=0)
        original = (self.out/'attempts.jsonl').read_bytes()
        with self.assertRaises(FileExistsError):
            measure(self.questions, self.url, self.out, pace_seconds=0)
        self.assertEqual((self.out/'attempts.jsonl').read_bytes(), original)
