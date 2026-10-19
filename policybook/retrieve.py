"""Stage 2: hybrid retrieval with role isolation and as_of eligibility.

Zero-leak rule: documents above the requester's role never enter the
candidate pool - both channels (BM25 and vector) are computed only over
chunks whose document the role may see. A separate blocked-evidence signal
(never part of the returned evidence) tells the service layer that matching
text exists above the role, which is the difference between refusing
"not_permitted" and lying "not_in_corpus" (rubric bar 6).

Temporal eligibility (freshness_contract.md): a document is eligible at
as_of when effective_date <= as_of and it is not superseded by another
corpus document whose effective_date <= as_of. A historical as_of can still
reach the older version; a superseded source is prohibited only when
presented as current, and retrieval at an as_of where it is superseded is
exactly that case.

Why in-process scoring instead of Chroma where-filters: at 391 passages the
whole index loads in milliseconds, both channels then share one identical
candidate set (no filter-semantics drift between stores), and the fused
ranking is plain numpy that fits in a review conversation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .bm25 import BM25
from .corpus import ROLES
from .db import COLLECTION
from .embed import MiniLMEncoder

ROLE_RANK = {role: i for i, role in enumerate(ROLES)}  # public < ops < legal
# Fusion: max-normalized weighted sum of the vector and BM25 channels.
# w_vec/w_lex tuned on the dev split only (see EXPERIMENT_LOG.md): BM25 is the
# stronger channel on regulatory text, the vector channel covers paraphrase.
W_VEC, W_LEX = 0.3, 0.7


@dataclass
class Hit:
    chunk_id: str
    section: str   # citation "section" (passage id)
    doc_id: str
    part: int
    text: str      # exact substring of source bytes
    score: float   # fused hybrid score in [0, 1]
    vec_rank: int
    bm25_rank: int


@dataclass
class Retrieval:
    hits: list
    blocked_evidence: bool  # above-role doc would have ranked without the filter
    allowed_pool: int       # chunks eligible after role + as_of filters
    top_cosine: float = 0.0   # raw best cosine over allowed pool (refusal signal)
    top_bm25: float = 0.0     # raw best BM25 over allowed pool (refusal signal)



def _fuse(idx, vec, lex):
    """Max-normalized weighted sum of channel scores over candidate idx."""
    lv = max((vec[i] for i in idx), default=0.0) or 1.0
    ll = max((lex[i] for i in idx), default=0.0) or 1.0
    return {i: W_VEC * vec[i] / lv + W_LEX * lex[i] / ll for i in idx}


class Retriever:
    def __init__(self, index_dir, model_dir):
        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(
            path=str(index_dir),
            settings=Settings(anonymized_telemetry=False))
        col = client.get_collection(COLLECTION)
        got = col.get(include=["documents", "metadatas", "embeddings"])
        if not got["ids"]:
            raise RuntimeError(f"index at {index_dir} is empty - run ingestion first")
        self.ids = list(got["ids"])
        self.texts = list(got["documents"])
        self.metas = list(got["metadatas"])
        self.embeddings = np.array(got["embeddings"], dtype=np.float32)
        self.bm25 = BM25(self.texts)
        self.encoder = MiniLMEncoder(model_dir)
        self.docs = {}
        for m in self.metas:
            d = m["doc_id"]
            if d not in self.docs:
                self.docs[d] = {
                    "role": m["role"],
                    "effective_date": m["effective_date"],
                    "supersedes": json.loads(m["supersedes"]),
                }

    def _eligible_docs(self, role: str, as_of: str):
        rank = ROLE_RANK[role]
        superseded = {old for m in self.docs.values()
                      if m["effective_date"] <= as_of for old in m["supersedes"]}
        allowed, above = set(), set()
        for d, m in self.docs.items():
            if ROLE_RANK[m["role"]] > rank:
                above.add(d)
            elif m["effective_date"] <= as_of and d not in superseded:
                allowed.add(d)
        return allowed, above

    def retrieve(self, question: str, role: str, as_of: str, k: int = 10) -> Retrieval:
        if role not in ROLE_RANK:
            raise ValueError(f"unknown role {role!r}")
        allowed, above = self._eligible_docs(role, as_of)
        allowed_idx = [i for i, m in enumerate(self.metas) if m["doc_id"] in allowed]

        q = self.encoder.encode([question])[0]
        vec = self.embeddings @ q  # L2-normalized at ingest: dot == cosine
        lex = self.bm25.scores(question)

        vec_order = sorted(allowed_idx, key=lambda i: (-vec[i], self.ids[i]))
        bm25_order = sorted(allowed_idx, key=lambda i: (-lex[i], self.ids[i]))
        vec_rank = {i: r for r, i in enumerate(vec_order, 1)}
        bm25_rank = {i: r for r, i in enumerate(bm25_order, 1)}
        fused = _fuse(allowed_idx, vec, lex)

        top = sorted(fused, key=lambda i: (-fused[i], self.ids[i]))[:k]
        hits = [Hit(chunk_id=self.ids[i], section=self.metas[i]["section"],
                    doc_id=self.metas[i]["doc_id"], part=self.metas[i]["part"],
                    text=self.texts[i], score=round(float(fused[i]), 6),
                    vec_rank=vec_rank[i], bm25_rank=bm25_rank[i])
                for i in top]

        # Blocked-evidence signal: would an above-role chunk have ranked in
        # the top k had the role filter not been applied? Scored on one
        # common scale over the union of allowed and role-denied (but
        # temporally eligible) chunks; the denied text never leaves this
        # signal - only the boolean does.
        blocked_idx = [i for i, m in enumerate(self.metas)
                       if m["doc_id"] in above and m["effective_date"] <= as_of]
        blocked = False
        if blocked_idx:
            fused_all = _fuse(allowed_idx + blocked_idx, vec, lex)
            top_all = sorted(fused_all, key=lambda i: (-fused_all[i], self.ids[i]))[:k]
            blocked = any(i in blocked_idx for i in top_all)

        top_cosine = float(max((vec[i] for i in allowed_idx), default=0.0))
        top_bm25 = float(max((lex[i] for i in allowed_idx), default=0.0))
        return Retrieval(hits=hits, blocked_evidence=blocked,
                         allowed_pool=len(allowed_idx),
                         top_cosine=round(top_cosine, 6),
                         top_bm25=round(top_bm25, 6))
