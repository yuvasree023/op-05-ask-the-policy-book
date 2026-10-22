"""Stage 3 tests: refusal policy, citation verbatim guarantee, role guard."""
import hashlib
import json
import os
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from policybook.answer import Answerer, NOT_IN_CORPUS_TEXT, NOT_PERMITTED_TEXT, norm, overlaps  # noqa: E402
from policybook.ingest import run as ingest_run  # noqa: E402

MODEL_DIR = REPO / "models" / "all-MiniLM-L6-v2"

DOCS = {
    "PUB_ANCHOR": ("public", "2020-01-01", [],
                   "[PUB_ANCHOR#s00000]\nOne third of the anchor investor portion is reserved "
                   "for domestic mutual funds.\n"),
    "PUB_KYC_OLD": ("public", "2019-01-01", [],
                    "[PUB_KYC_OLD#s00000]\nMinors open accounts with a guardian jointly.\n"),
    "PUB_KYC_NEW": ("public", "2021-06-01", ["PUB_KYC_OLD"],
                    "[PUB_KYC_NEW#s00000]\nA minor above ten years operates the account independently.\n"),
    "LEGAL_RULE": ("legal", "2022-01-01", [],
                   "[LEGAL_RULE#s00000]\nPrivileged litigation hold procedure for counsel only, "
                   "covering retention of every client communication record for seven full years.\n"),
}


def write_corpus(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "text").mkdir(exist_ok=True)
    manifest = []
    for doc_id, (role, eff, sup, body) in DOCS.items():
        path = f"text/{doc_id}.txt"
        (root / path).write_text(body)
        manifest.append({"doc_id": doc_id, "title": doc_id,
                         "source_url": "https://example.test/src",
                         "effective_date": eff, "supersedes": sup,
                         "role": role, "text_path": path,
                         "sha256": hashlib.sha256(body.encode()).hexdigest()})
    (root / "manifest.jsonl").write_text(
        "\n".join(json.dumps(m) for m in manifest) + "\n")


class AnswerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        root = Path(cls._td.name)
        write_corpus(root / "corpus")
        assert ingest_run(["--corpus-dir", str(root / "corpus"),
                           "--index-dir", str(root / "index"),
                           "--report", str(root / "report.json")]) == 0
        cls.ans = Answerer(root / "index", MODEL_DIR, root / "corpus")
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def test_role_blocked_refuses_not_permitted(self):
        r = self.ans.answer("q1", "What is the privileged litigation hold procedure?",
                            "public", "2023-01-01")
        self.assertEqual(r.status, "not_permitted")
        self.assertEqual(r.citations, [])
        self.assertNotIn("LEGAL_RULE", r.answer)
        self.assertNotIn("litigation hold procedure for counsel", r.answer.lower())
        self.assertNotIn("retention of every client communication", r.answer.lower())

    def test_out_of_corpus_refusal(self):
        # role legal: no above-role docs exist, so this exercises the pure
        # not_in_corpus path (tiny fixture corpora make the blocked signal
        # trivially true for lower roles - every chunk is "top-k").
        r = self.ans.answer("q2", "What is the airspeed velocity of an unladen swallow?",
                            "legal", "2023-01-01")
        self.assertEqual(r.status, "not_in_corpus")
        self.assertEqual(r.citations, [])

    def test_answered_quote_verbatim_and_allowed(self):
        r = self.ans.answer("q3", "What fraction of the anchor investor portion "
                                  "is reserved for domestic mutual funds?",
                            "legal", "2023-01-01")
        self.assertEqual(r.status, "answered")
        self.assertEqual(len(r.citations), 1)
        c = r.citations[0]
        self.assertEqual(c.doc_id, "PUB_ANCHOR")
        text = (self.root / "corpus" / "text" / "PUB_ANCHOR.txt").read_text()
        self.assertIn(norm(c.quote), norm(text))
        self.assertIn(c.quote, text)  # exact substring, not just normalized

    def test_role_guard_rejects_above_role_overlap(self):
        # A 12-token run from the legal doc must trip the guard for public...
        legal_text = (self.root / "corpus" / "text" / "LEGAL_RULE.txt").read_text()
        leak = ("Privileged litigation hold procedure for counsel only, covering "
                "retention of every client communication record for seven full years.")
        self.assertTrue(overlaps(norm(leak), norm(legal_text), 12))
        self.assertFalse(self.ans._role_safe("public", leak, []))
        # ...and pass for legal, who may see it.
        self.assertTrue(self.ans._role_safe("legal", leak, []))

    def test_answer_has_no_above_role_overlap(self):
        r = self.ans.answer("q4", "What fraction of the anchor investor portion "
                                  "is reserved for domestic mutual funds?",
                            "public", "2023-01-01")
        if r.status == "answered":
            legal_text = (self.root / "corpus" / "text" / "LEGAL_RULE.txt").read_text()
            self.assertFalse(overlaps(norm(r.answer), norm(legal_text), 12))

    def test_temporal_current_vs_historical(self):
        cur = self.ans.answer("q5", "How does a minor operate the account?",
                              "legal", "2023-01-01")
        self.assertEqual(cur.status, "answered")
        self.assertEqual(cur.citations[0].doc_id, "PUB_KYC_NEW")
        hist = self.ans.answer("q6", "How does a minor operate the account?",
                               "legal", "2020-01-01")
        self.assertEqual(hist.status, "answered")
        self.assertEqual(hist.citations[0].doc_id, "PUB_KYC_OLD")
        self.assertIn("later version", hist.answer)  # future change disclosed

    def test_backend_auto_activates_on_gateway_credentials(self):
        # Approved 2026-10-22: reviewer gateway credentials (LLM_BASE_URL +
        # LLM_API_KEY) activate the model generator with no OP05-specific
        # flag; LLM_MODEL supplies the model name when present.
        with mock.patch.dict(os.environ, {"LLM_BASE_URL": "http://127.0.0.1:9",
                                          "LLM_API_KEY": "k", "LLM_MODEL": "m-x"}):
            os.environ.pop("OP05_GEN_BACKEND", None)
            a = Answerer(self.root / "index", MODEL_DIR, self.root / "corpus")
            self.assertEqual(a.gen_backend, "openai")
            self.assertTrue(a.gen_auto)
            self.assertEqual(a.gen_model, "m-x")

    def test_explicit_backend_beats_gateway_credentials(self):
        with mock.patch.dict(os.environ, {"LLM_BASE_URL": "http://127.0.0.1:9",
                                          "LLM_API_KEY": "k",
                                          "OP05_GEN_BACKEND": "extractive"}):
            a = Answerer(self.root / "index", MODEL_DIR, self.root / "corpus")
            self.assertEqual(a.gen_backend, "extractive")
            self.assertFalse(a.gen_auto)

    def test_extractive_default_without_credentials(self):
        with mock.patch.dict(os.environ, {}):
            for k in ("LLM_BASE_URL", "LLM_API_KEY", "OP05_GEN_BACKEND"):
                os.environ.pop(k, None)
            a = Answerer(self.root / "index", MODEL_DIR, self.root / "corpus")
            self.assertEqual(a.gen_backend, "extractive")
            self.assertFalse(a.gen_auto)

    def test_refusal_texts_name_no_corpus_doc(self):
        for doc_id in DOCS:
            self.assertNotIn(doc_id, NOT_PERMITTED_TEXT)
            self.assertNotIn(doc_id, NOT_IN_CORPUS_TEXT)


if __name__ == "__main__":
    unittest.main()
