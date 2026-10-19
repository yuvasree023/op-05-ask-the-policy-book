"""Stage 1 tests: synthetic corpus end to end, idempotency, swap detection."""
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from policybook.corpus import CorpusError, read_corpus  # noqa: E402
from policybook.ingest import run as ingest_run  # noqa: E402

MODEL_DIR = REPO / "models" / "all-MiniLM-L6-v2"
LONG_BODY = ("The board shall review the exposure every quarter. " * 60)  # ~3000 chars


def write_corpus(root: Path, long_body: str = LONG_BODY) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "text").mkdir(exist_ok=True)
    texts = {
        "DOC_A": ("[DOC_A#s00000]\nShort opening rule.\n\n"
                  f"[DOC_A#s00001]\n{long_body}\n"),
        "DOC_B": "[DOC_B#s00000]\nRestricted internal rule.\n",
    }
    manifest = []
    passages = []
    for doc_id, body in texts.items():
        path = f"text/{doc_id}.txt"
        (root / path).write_text(body)
        manifest.append({
            "doc_id": doc_id, "title": f"Doc {doc_id}",
            "source_url": "https://example.test/src",
            "effective_date": "2026-01-01", "supersedes": [],
            "role": {"DOC_A": "public", "DOC_B": "legal"}[doc_id],
            "text_path": path,
            "sha256": hashlib.sha256(body.encode()).hexdigest()})
        # passages.jsonl mirrors the section parsing
        import re as _re
        parts = _re.split(r"^\[(.+)\]\s*$", body, flags=_re.M)
        for i in range(1, len(parts), 2):
            passages.append({"passage_id": parts[i], "doc_id": doc_id,
                             "text": parts[i + 1].strip("\n")})
    (root / "manifest.jsonl").write_text(
        "\n".join(json.dumps(m) for m in manifest) + "\n")
    (root / "passages.jsonl").write_text(
        "\n".join(json.dumps(p) for p in passages) + "\n")


class CorpusTests(unittest.TestCase):
    def test_read_corpus_parses_sections_and_roles(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "corpus"
            write_corpus(root)
            corpus = read_corpus(root)
            self.assertEqual(len(corpus.docs), 2)
            self.assertEqual(len(corpus.sections), 3)
            self.assertEqual(corpus.docs["DOC_B"].role, "legal")
            self.assertEqual(corpus.validation, [])
            ids = [s.passage_id for s in corpus.sections]
            self.assertIn("DOC_A#s00001", ids)

    def test_sha_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "corpus"
            write_corpus(root)
            with open(root / "text" / "DOC_A.txt", "a") as f:
                f.write("tampered\n")
            with self.assertRaises(CorpusError):
                read_corpus(root)

    def test_missing_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "corpus"
            root.mkdir()
            (root / "random.txt").write_text("hello")
            with self.assertRaises(CorpusError):
                read_corpus(root)


class IngestTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.corpus = Path(self.td) / "corpus"
        self.index = Path(self.td) / ".index"
        self.report = Path(self.td) / "report.json"
        write_corpus(self.corpus)

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def _ingest(self, *extra):
        rc = ingest_run(["--corpus-dir", str(self.corpus),
                         "--index-dir", str(self.index),
                         "--model-dir", str(MODEL_DIR),
                         "--report", str(self.report), *extra])
        self.assertEqual(rc, 0)
        return json.loads(self.report.read_text())

    def test_full_cycle(self):
        first = self._ingest()
        self.assertFalse(first["skipped"])
        # s00001 of DOC_A is ~3000 chars and must split into several chunks
        self.assertGreater(first["chunks"], 3)
        self.assertEqual(first["sections"], 3)
        self.assertEqual(first["embedding_dim"], 384)
        self.assertEqual(first["validation"], [])

        # metadata survives: role is what later stages filter on
        sys.path.insert(0, str(REPO))
        from policybook.db import VectorStore
        store = VectorStore(self.index)
        self.assertEqual(store.count(), first["chunks"])
        col = store.client.get_collection("passages")
        got = col.get(where={"doc_id": "DOC_B"}, include=["metadatas"])
        self.assertEqual(got["metadatas"][0]["role"], "legal")
        self.assertEqual(got["metadatas"][0]["effective_date"], "2026-01-01")

        # unchanged corpus: skip, no re-embedding
        second = self._ingest()
        self.assertTrue(second["skipped"])

        # modified corpus: hash changes, index rebuilds
        body = (self.corpus / "text" / "DOC_A.txt").read_text() + (
            "[DOC_A#s00002]\nA brand new section.\n")
        (self.corpus / "text" / "DOC_A.txt").write_text(body)
        lines = (self.corpus / "manifest.jsonl").read_text().splitlines()
        updated = []
        for line in lines:
            rec = json.loads(line)
            if rec["doc_id"] == "DOC_A":
                rec["sha256"] = hashlib.sha256(body.encode()).hexdigest()
            updated.append(json.dumps(rec))
        (self.corpus / "manifest.jsonl").write_text("\n".join(updated) + "\n")
        third = self._ingest()
        self.assertFalse(third["skipped"])
        self.assertEqual(third["sections"], 4)
        self.assertNotEqual(third["corpus_hash"], first["corpus_hash"])


if __name__ == "__main__":
    unittest.main()
