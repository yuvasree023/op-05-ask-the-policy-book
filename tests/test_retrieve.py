"""Stage 2 tests: role isolation, as_of eligibility, supersession, fusion."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from policybook.ingest import run as ingest_run  # noqa: E402
from policybook.retrieve import ROLE_RANK, Retriever  # noqa: E402

MODEL_DIR = REPO / "models" / "all-MiniLM-L6-v2"

DOCS = {
    # doc_id: (role, effective_date, supersedes, text)
    "PUB_KYC_OLD": ("public", "2019-01-01", [],
                    "[PUB_KYC_OLD#s00000]\nMinors open accounts with a guardian jointly.\n"),
    "PUB_KYC_NEW": ("public", "2021-06-01", ["PUB_KYC_OLD"],
                    "[PUB_KYC_NEW#s00000]\nA minor above ten years operates the account independently.\n"),
    "PUB_ANCHOR": ("public", "2020-01-01", [],
                   "[PUB_ANCHOR#s00000]\nOne third of the anchor investor portion is reserved for domestic mutual funds.\n"),
    "OPS_RULE": ("ops", "2021-06-01", [],
                 "[OPS_RULE#s00000]\nBranch staff must log every cash deposit above ten lakh same day.\n"),
    "LEGAL_RULE": ("legal", "2022-01-01", [],
                   "[LEGAL_RULE#s00000]\nPrivileged litigation hold procedure for counsel only.\n"),
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


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        root = Path(cls._td.name)
        write_corpus(root / "corpus")
        rc = ingest_run(["--corpus-dir", str(root / "corpus"),
                         "--index-dir", str(root / "index"),
                         "--report", str(root / "report.json")])
        assert rc == 0
        cls.retr = Retriever(root / "index", MODEL_DIR)

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def test_role_isolation_public(self):
        r = self.retr.retrieve("litigation hold procedure", "public", "2023-01-01")
        self.assertTrue(all(h.doc_id != "LEGAL_RULE" for h in r.hits))
        self.assertTrue(all(ROLE_RANK[DOCS[h.doc_id][0]] <= 0 for h in r.hits))
        self.assertTrue(r.blocked_evidence)  # the legal doc exists but is denied

    def test_role_isolation_ops_cannot_see_legal(self):
        r = self.retr.retrieve("litigation hold procedure", "ops", "2023-01-01")
        self.assertTrue(all(h.doc_id != "LEGAL_RULE" for h in r.hits))
        self.assertTrue(r.blocked_evidence)

    def test_legal_sees_everything(self):
        r = self.retr.retrieve("litigation hold procedure", "legal", "2023-01-01")
        self.assertIn("LEGAL_RULE", {h.doc_id for h in r.hits})
        self.assertFalse(r.blocked_evidence)

    def test_temporal_exclusion(self):
        # OPS_RULE effective 2021-06-01; an as_of before that must not see it
        r = self.retr.retrieve("cash deposit logging", "ops", "2020-01-01")
        self.assertNotIn("OPS_RULE", {h.doc_id for h in r.hits})
        r2 = self.retr.retrieve("cash deposit logging", "ops", "2022-01-01")
        self.assertIn("OPS_RULE", {h.doc_id for h in r2.hits})

    def test_supersession(self):
        # before the new rule: old rule retrievable, new one not yet effective
        r = self.retr.retrieve("minor account guardian", "public", "2020-01-01")
        self.assertIn("PUB_KYC_OLD", {h.doc_id for h in r.hits})
        self.assertNotIn("PUB_KYC_NEW", {h.doc_id for h in r.hits})
        # after: new rule governs, superseded old rule excluded
        r2 = self.retr.retrieve("minor account guardian", "public", "2022-01-01")
        self.assertIn("PUB_KYC_NEW", {h.doc_id for h in r2.hits})
        self.assertNotIn("PUB_KYC_OLD", {h.doc_id for h in r2.hits})

    def test_lexical_and_semantic_queries(self):
        r = self.retr.retrieve("anchor investor mutual funds fraction", "legal", "2023-01-01")
        self.assertEqual(r.hits[0].doc_id, "PUB_ANCHOR")
        r2 = self.retr.retrieve("What part of the anchor book goes to MFs?", "legal", "2023-01-01")
        self.assertIn("PUB_ANCHOR", {h.doc_id for h in r2.hits[:3]})


if __name__ == "__main__":
    unittest.main()
