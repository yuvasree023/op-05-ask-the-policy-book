"""In-process passage store: corpus-hash-keyed cache, no vector database.

Build spec (user approach, 2026-10-19, step 2): 391 passages, brute force is
fine, no vector DB. Embeddings, chunk text and metadata are cached as one
local npz+json pair under the index dir. The json sidecar carries the corpus
hash: a swapped corpus (freshness update pack) hashes differently and forces
a rebuild, and an unchanged corpus skips re-embedding entirely. Retrieval
loads the whole cache in milliseconds and scores in plain numpy, so both
channels always share one identical candidate set.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

STORE_NPZ = "passages.npz"
STORE_JSON = "passages.json"


def _paths(index_dir):
    d = Path(index_dir)
    return d / STORE_NPZ, d / STORE_JSON


class PassageStore:
    """File-backed cache of chunks + embeddings + role/time metadata."""

    def __init__(self, index_dir):
        self.index_dir = Path(index_dir)

    def stored_corpus_hash(self):
        _, jp = _paths(self.index_dir)
        if not jp.exists():
            return None
        try:
            return json.loads(jp.read_text()).get("corpus_hash")
        except Exception:
            return None

    def count(self) -> int:
        _, jp = _paths(self.index_dir)
        if not jp.exists():
            return 0
        try:
            return len(json.loads(jp.read_text()).get("ids", []))
        except Exception:
            return 0

    def rebuild(self, chunks, embeddings, corpus, model_name: str):
        self.index_dir.mkdir(parents=True, exist_ok=True)
        npz_path, jp = _paths(self.index_dir)
        metas = [{
            "doc_id": c.doc_id,
            "section": c.passage_id,
            "part": c.part,
            "role": corpus.docs[c.doc_id].role,
            "effective_date": corpus.docs[c.doc_id].effective_date,
            "supersedes": json.dumps(corpus.docs[c.doc_id].supersedes),
            "doc_sha256": corpus.docs[c.doc_id].sha256,
            "chunk_chars": len(c.text),
        } for c in chunks]
        jp.write_text(json.dumps({
            "corpus_hash": corpus.corpus_hash,
            "embedding_model": model_name,
            "documents": len(corpus.docs),
            "sections": len(corpus.sections),
            "ids": [c.chunk_id for c in chunks],
            "texts": [c.text for c in chunks],
            "metas": metas,
        }))
        np.savez_compressed(npz_path,
                            embeddings=np.asarray(embeddings, dtype=np.float32))

    def load(self):
        """Return (ids, texts, metas, embeddings); raise when no index."""
        npz_path, jp = _paths(self.index_dir)
        if not jp.exists() or not npz_path.exists():
            raise RuntimeError(
                f"index at {self.index_dir} is empty - run ingestion first")
        meta = json.loads(jp.read_text())
        if not meta.get("ids"):
            raise RuntimeError(
                f"index at {self.index_dir} is empty - run ingestion first")
        embeddings = np.load(npz_path)["embeddings"]
        return meta["ids"], meta["texts"], meta["metas"], embeddings
