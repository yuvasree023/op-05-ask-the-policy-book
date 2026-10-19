"""Stage 1 entry point: corpus dir -> validate -> chunk -> embed -> cached store.

Usage:
    python -m policybook.ingest --corpus-dir /path/to/corpus [--index-dir .index]
                                [--model-dir models/all-MiniLM-L6-v2] [--force]

Idempotent: an unchanged corpus (same hash) skips re-embedding; a swapped or
modified corpus rebuilds the index from scratch. Prints a JSON summary and
writes results/ingest_report.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .chunk import chunk_section
from .corpus import CorpusError, read_corpus
from .db import PassageStore
from .embed import MODEL_NAME, MiniLMEncoder

REPO_ROOT = Path(__file__).resolve().parent.parent


def run(argv=None) -> int:
    p = argparse.ArgumentParser(prog="policybook.ingest")
    p.add_argument("--corpus-dir", default=None,
                   help="corpus directory (default: $CORPUS_DIR or ./data/corpus)")
    p.add_argument("--index-dir", default=None,
                   help="index cache directory (default: $INDEX_DIR or ./.index)")
    p.add_argument("--model-dir", default=str(REPO_ROOT / "models" / "all-MiniLM-L6-v2"))
    p.add_argument("--report", default=str(REPO_ROOT / "results" / "ingest_report.json"))
    p.add_argument("--force", action="store_true", help="rebuild even if corpus unchanged")
    args = p.parse_args(argv)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)

    import os
    corpus_dir = args.corpus_dir or os.environ.get("CORPUS_DIR") or str(REPO_ROOT / "data" / "corpus")
    index_dir = args.index_dir or os.environ.get("INDEX_DIR") or str(REPO_ROOT / ".index")

    t0 = time.perf_counter()
    try:
        corpus = read_corpus(corpus_dir)
    except CorpusError as e:
        print(f"ingest: corpus error: {e}", file=sys.stderr)
        return 2

    store = PassageStore(index_dir)
    report = {
        "corpus_dir": str(Path(corpus_dir).resolve()),
        "index_dir": str(Path(index_dir).resolve()),
        "corpus_hash": corpus.corpus_hash,
        "documents": len(corpus.docs),
        "sections": len(corpus.sections),
        "validation": corpus.validation,
        "embedding_model": MODEL_NAME,
    }

    if not args.force and store.stored_corpus_hash() == corpus.corpus_hash and store.count() > 0:
        report.update({"skipped": True, "reason": "corpus unchanged",
                       "chunks": store.count(),
                       "seconds": round(time.perf_counter() - t0, 3)})
        Path(args.report).write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return 0

    chunks = [c for s in corpus.sections for c in chunk_section(s)]
    n_split = len({c.passage_id for c in chunks} - {c.chunk_id for c in chunks})

    t1 = time.perf_counter()
    encoder = MiniLMEncoder(args.model_dir)
    embeddings = encoder.encode([c.text for c in chunks])
    embed_s = time.perf_counter() - t1

    t2 = time.perf_counter()
    store.rebuild(chunks, embeddings, corpus, MODEL_NAME)
    index_s = time.perf_counter() - t2

    assert store.count() == len(chunks), "store count mismatch after rebuild"

    report.update({
        "skipped": False,
        "chunks": len(chunks),
        "sections_split": n_split,
        "embedding_dim": int(embeddings.shape[1]),
        "embed_seconds": round(embed_s, 3),
        "index_seconds": round(index_s, 3),
        "seconds": round(time.perf_counter() - t0, 3),
    })
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
