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
from .db import PassageStore
from .embed import MiniLMEncoder

ROLE_RANK = {role: i for i, role in enumerate(ROLES)}  # public < ops < legal
# Fusion: reciprocal rank fusion (RRF) over the vector and BM25 channels
# (spec: user-pasted approach 2026-10-19, step 2). Rank-based fusion needs no
# per-channel score normalization and no tuned weights; k=60 is the standard
# constant (Cormack et al. 2009). Supersedes the v1 max-normalized weighted
# sum (0.3 vector / 0.7 BM25) - v1 metrics stay labeled, never relabeled.
RRF_K = 60


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
    """Reciprocal rank fusion of the two channels over candidate idx.

    Ranks are computed within idx (1 = best), so the same function serves the
    allowed pool and the allowed+denied union used for the blocked signal.
    """
    vec_order = sorted(idx, key=lambda i: (-vec[i], i))
    lex_order = sorted(idx, key=lambda i: (-lex[i], i))
    vec_rank = {i: r for r, i in enumerate(vec_order, 1)}
    lex_rank = {i: r for r, i in enumerate(lex_order, 1)}
    return {i: 1.0 / (RRF_K + vec_rank[i]) + 1.0 / (RRF_K + lex_rank[i])
            for i in idx}


class Retriever:
    def __init__(self, index_dir, model_dir):
        ids, texts, metas, embeddings = PassageStore(index_dir).load()
        self.ids = list(ids)
        self.texts = list(texts)
        self.metas = list(metas)
        self.embeddings = np.array(embeddings, dtype=np.float32)
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
