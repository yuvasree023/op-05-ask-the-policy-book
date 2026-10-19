"""Chroma vector store: persistent, local, metadata-filterable.

Metadata filtering is what later stages use for role isolation and as_of
(temporal) filtering, so role / effective_date / supersedes are stored on
every chunk. The collection metadata carries the corpus hash: a swapped
corpus (freshness update pack) hashes differently and forces a rebuild, and
an unchanged corpus skips re-embedding entirely.
"""
from __future__ import annotations

import json

import chromadb
from chromadb.config import Settings

COLLECTION = "passages"


class VectorStore:
    def __init__(self, index_dir):
        # anonymized_telemetry=False: no posthog phone-home. The reviewer
        # sandbox meters egress, and "offline ingestion" must mean it.
        self.client = chromadb.PersistentClient(
            path=str(index_dir),
            settings=Settings(anonymized_telemetry=False))

    def _get(self):
        try:
            return self.client.get_collection(COLLECTION)
        except Exception:
            return None

    def stored_corpus_hash(self):
        col = self._get()
        return (col.metadata or {}).get("corpus_hash") if col else None

    def count(self) -> int:
        col = self._get()
        return col.count() if col else 0

    def rebuild(self, chunks, embeddings, corpus, model_name: str):
        if self._get() is not None:
            self.client.delete_collection(COLLECTION)
        col = self.client.create_collection(
            COLLECTION,
            metadata={"hnsw:space": "cosine", "corpus_hash": corpus.corpus_hash,
                      "embedding_model": model_name,
                      "documents": len(corpus.docs),
                      "sections": len(corpus.sections)})
        for i in range(0, len(chunks), 256):
            batch = chunks[i:i + 256]
            col.add(
                ids=[c.chunk_id for c in batch],
                embeddings=[e.tolist() for e in embeddings[i:i + 256]],
                documents=[c.text for c in batch],
                metadatas=[{
                    "doc_id": c.doc_id,
                    "section": c.passage_id,
                    "part": c.part,
                    "role": corpus.docs[c.doc_id].role,
                    "effective_date": corpus.docs[c.doc_id].effective_date,
                    "supersedes": json.dumps(corpus.docs[c.doc_id].supersedes),
                    "doc_sha256": corpus.docs[c.doc_id].sha256,
                    "chunk_chars": len(c.text),
                } for c in batch])
        return col
