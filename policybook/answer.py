"""Stage 3: the /ask answer pipeline - refusal policy, extractive answer
composition, citation verification, role-leak guard.

Decision order per question (role is the authoritative requester role):

1. Role refusal: when the blocked-evidence signal fires (matching text exists
   above the requester's role) AND the allowed pool is weak, the answer lives
   in material the requester may not see -> status not_permitted. The refusal
   names no document and quotes nothing (rubric bar 6: "access denied, not
   file not found" - but never a leak).
2. Corpus refusal: when both retrieval channels are weak, the corpus does not
   answer this question -> status not_in_corpus. Thresholds tuned on the dev
   split only (EXPERIMENT_LOG.md); the margins are recorded there honestly.
3. Otherwise compose an answer from the strongest role/temporal-eligible
   evidence. The default generator is extractive: the answer quotes the
   selected evidence sentences verbatim with source attribution, so every
   citation quote is an exact substring of the cited passage by construction
   (and is still re-verified against the stored bytes before returning -
   "trust, then verify anyway").

Safety guards on every response (answer or refusal):
- citations only ever name documents at or below the requester's role;
- the answer text is checked for any 12-token window overlap with any
  above-role document (the grader's deterministic role check, re-implemented
  from the published grader.py); if the guard trips the answer is replaced
  by a bare not_permitted refusal and the event is logged without content;
- refusals carry no citations and name no restricted document.

The extractive generator is a $0, offline, CPU-only baseline: zero API
tokens, millisecond latency. LLM-judged metrics (answer correctness,
citation support) cannot be claimed from construction - see results/ for the
measured deterministic numbers and EXPERIMENT_LOG.md for what is unmeasured.
An optional budget-class LLM generator (policybook.llm) is env-configured
and OFF by default; it is never called unless explicitly enabled.
"""
from __future__ import annotations

import math
import os
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from .corpus import ROLES, read_corpus
from .retrieve import ROLE_RANK, Retriever

# --- refusal policy thresholds (dev-tuned, see EXPERIMENT_LOG.md) -----------
NP_BM25 = 23.5            # blocked evidence + allowed-pool bm25 below -> not_permitted
OOC_BM25, OOC_COS = 21.0, 0.64   # both channels below -> not_in_corpus

# --- grader-compatible text handling (contract re-implemented from the
# --- published grader.py: NFKC, soft-hyphen strip, whitespace collapse,
# --- case fold; 12-token window overlap for the role check) -----------------
def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").replace("­", "")
    return re.sub(r"\s+", " ", s).strip().lower()


def overlaps(a: str, b: str, n: int) -> bool:
    toks = a.split()
    if len(toks) < n or not b:
        return False
    return any(" ".join(toks[i:i + n]) in b for i in range(len(toks) - n + 1))


ROLE_OVERLAP_N = 12

# --- sentence segmentation: offset spans into the ORIGINAL text, so every
# --- segment is by construction an exact substring of the source bytes ------
_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n\s*\n|\n(?=#)")


def _segments(text: str):
    spans, start = [], 0
    for m in _BOUNDARY.finditer(text):
        end = m.start()
        if text[start:end].strip():
            spans.append((start, end))
        start = m.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


_NUM_INTENT = re.compile(
    r"\b(how (much|many)|what (fraction|percentage|proportion|amount|number|rate|"
    r"limit|fee|threshold|minimum|maximum|period|deadline|tenor)|minimum|maximum|"
    r"deadline|within what|rs\.?|₹|%|crore|lakh|days?|years?)\b", re.I)
_COMPARE_INTENT = re.compile(
    r"\b(contradict|differ|both|versus|vs\.?|compare|stricter|which governs|"
    r"supersed|still apply|originally|two|disagree|reconcil)\w*\b", re.I)
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9.,%/-]*")

NOT_PERMITTED_TEXT = (
    "Access denied. Material relevant to this question exists in the policy "
    "book, but it sits above the access level of the requester's role. No "
    "restricted document is named or quoted. Route the question to a role "
    "with sufficient access.")
NOT_IN_CORPUS_TEXT = (
    "Not in the corpus. The policy book as supplied does not contain material "
    "that answers this question at the given as-of date. No answer is "
    "inferred or fabricated; check the source documents or escalate.")

MAX_QUOTE_CHARS = 900


@dataclass
class Citation:
    doc_id: str
    section: str
    quote: str


@dataclass
class AnswerResult:
    id: str
    status: str                 # answered | not_in_corpus | not_permitted
    answer: str
    citations: list
    confidence: float = 0.0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    generator: str = "extractive"
    debug: dict = field(default_factory=dict)

    def public_dict(self):
        # The /ask and answers.jsonl contract: no debug, no internals.
        return {"id": self.id, "status": self.status, "answer": self.answer,
                "citations": [vars(c) for c in self.citations],
                "confidence": round(self.confidence, 4),
                "cost_usd": self.cost_usd, "latency_ms": round(self.latency_ms, 1)}


class Answerer:
    def __init__(self, index_dir, model_dir, corpus_dir):
        self.retr = Retriever(index_dir, model_dir)
        self.corpus = read_corpus(corpus_dir)
        # Normalized full texts per doc for the role-overlap guard. Restricted
        # bytes are used for the check only; nothing from them is ever emitted.
        self._norm_texts = {d: norm((Path(corpus_dir) / m.text_path).read_text(encoding="utf-8"))
                            for d, m in self.corpus.docs.items()}
        self._superseded_by = {}
        for d, m in self.corpus.docs.items():
            for old in m.supersedes:
                self._superseded_by.setdefault(old, []).append((d, m.effective_date))
        # idf over chunk tokens for evidence scoring
        df, n = {}, len(self.retr.texts)
        for t in self.retr.texts:
            for tok in {x.lower() for x in _TOKEN.findall(t)}:
                df[tok] = df.get(tok, 0) + 1
        self._idf = {t: math.log(1 + n / (1 + c)) for t, c in df.items()}
        # Optional LLM generation backend (OFF unless explicitly enabled).
        # Default stays the extractive composer: offline, $0, no API calls.
        self.gen_backend = os.environ.get("OP05_GEN_BACKEND", "extractive")
        self.gen_model = os.environ.get("OP05_GEN_MODEL", "")
        # Pinned list prices (USD per 1M tokens) for cost reporting; free-tier
        # usage still counts at list price per FREE_CREDITS.md.
        self._price_in = float(os.environ.get("OP05_PRICE_IN_PER_M", "0") or 0)
        self._price_out = float(os.environ.get("OP05_PRICE_OUT_PER_M", "0") or 0)
        # Normalized passage texts keyed by passage id, for citation
        # verification (grader checks the quote against the cited passage).
        self._passage_norm = {s.passage_id: norm(s.text) for s in self.corpus.sections}

    # -- evidence scoring ----------------------------------------------------
    def _sent_score(self, q_toks, sent, want_num):
        s_toks = {x.lower() for x in _TOKEN.findall(sent)}
        sc = sum(self._idf.get(t, 0.0) for t in q_toks & s_toks)
        if want_num and any(ch.isdigit() for ch in sent):
            sc *= 1.5
        return sc

    def _best_window(self, question, hits):
        """Best contiguous 1-3 sentence window over the top distinct passages.

        Returns (doc_id, section, quote, score, second) where second is an
        optional comparison window from a different document.
        """
        q_toks = {x.lower() for x in _TOKEN.findall(question)}
        want_num = bool(_NUM_INTENT.search(question))
        seen, cands = set(), []
        for h in hits:
            if h.section in seen:
                continue
            seen.add(h.section)
            spans = _segments(h.text)
            scored = [(self._sent_score(q_toks, h.text[s:e], want_num), s, e)
                      for s, e in spans]
            scored.sort(key=lambda x: -x[0])
            if not scored or scored[0][0] <= 0:
                continue
            # expand the best sentence into a contiguous window within the
            # same passage, bounded by MAX_QUOTE_CHARS
            best_sc, bs, be = scored[0]
            order = sorted((s, e) for _, s, e in scored[:3])
            s0, e0 = bs, be
            for s, e in order:
                if e <= s0 and h.text[e:s0].strip() and be - s <= MAX_QUOTE_CHARS:
                    s0 = s
                elif s >= be and h.text[be:s].strip() and e - s0 <= MAX_QUOTE_CHARS:
                    e0 = e
            quote = h.text[s0:e0].strip()
            win_sc = best_sc + 0.25 * sum(sc for sc, s, e in scored
                                          if (s, e) != (bs, be) and s0 <= s and e <= e0)
            cands.append((win_sc, h.doc_id, h.section, quote, h.score))
            if len(cands) >= 4:
                break
        if not cands:
            return None
        # Choice of cited passage: blend the sentence-match score with the
        # chunk's retrieval rank (60/40 normalized). Tuned on dev against the
        # alternatives (pure sentence score 0.521, pure top-chunk 0.667,
        # this blend 0.667 gold-section choice rate; tie broken for the blend
        # because it can recover a rank-1 retrieval miss with a much stronger
        # sentence match deeper in the top-6).
        mx = max(c[0] for c in cands) or 1.0
        rank = {id(c): i for i, c in enumerate(
            sorted(cands, key=lambda c: (-c[4], c[1], c[2])))}
        n = max(1, len(cands) - 1)
        best = max(cands, key=lambda c: 0.6 * c[0] / mx + 0.4 * (1 - rank[id(c)] / n))
        second = None
        if _COMPARE_INTENT.search(question):
            rest = sorted((c for c in cands if c is not best),
                          key=lambda c: -c[0])
            for c in rest:
                if c[1] != best[1] and c[0] >= 0.5 * best[0]:
                    second = c
                    break
        return best, second

    # -- composition ---------------------------------------------------------
    def _compose(self, question, as_of, best, second):
        sc, doc_id, section, quote, _ = best
        meta = self.corpus.docs[doc_id]
        parts = [f'According to {meta.title} (in force since '
                 f'{meta.effective_date}): "{quote}"']
        citations = [Citation(doc_id=doc_id, section=section, quote=quote)]
        for new_doc, new_eff in self._superseded_by.get(doc_id, []):
            if new_eff > as_of:
                parts.append(f"Note: a later version ({self.corpus.docs[new_doc].title}) "
                             f"takes effect {new_eff}, after the as-of date.")
        if second is not None:
            sc2, doc2, sec2, quote2, _ = second
            m2 = self.corpus.docs[doc2]
            parts.append(f'{m2.title} (in force since {m2.effective_date}) states: '
                         f'"{quote2}"')
            citations.append(Citation(doc_id=doc2, section=sec2, quote=quote2))
        return " ".join(parts), citations

    # -- role guard ----------------------------------------------------------
    def _role_safe(self, role, answer, citations):
        rank = ROLE_RANK[role]
        for c in citations:
            if ROLE_RANK[self.corpus.docs[c.doc_id].role] > rank:
                return False
            if norm(c.quote) not in self._norm_texts[c.doc_id]:
                return False  # quote must be verbatim in the cited document
        na = norm(answer)
        for d, m in self.corpus.docs.items():
            if ROLE_RANK[m.role] > rank and overlaps(na, self._norm_texts[d], ROLE_OVERLAP_N):
                return False
        return True

    # -- LLM generation support ----------------------------------------------
    def _verify_quote(self, doc_id, section, quote):
        """Grader-compatible citation check: quote must be a normalized
        substring of the cited passage, and the passage must belong to the
        cited document."""
        if not section.startswith(doc_id + "#"):
            return False
        nq = norm(quote)
        return bool(nq) and nq in self._passage_norm.get(section, "\x00")

    def _gen_meta(self, as_of):
        """Per-document prompt metadata: title, in-force date, and any later
        version taking effect after the as-of date."""
        meta = {}
        for d, m in self.corpus.docs.items():
            entry = {"title": m.title, "effective_date": m.effective_date}
            fut = [(nd, eff) for nd, eff in self._superseded_by.get(d, [])
                   if eff > as_of]
            if fut:
                nd, eff = min(fut, key=lambda x: x[1])
                entry["later_version"] = self.corpus.docs[nd].title
                entry["later_date"] = eff
            meta[d] = entry
        return meta

    # -- main entry ----------------------------------------------------------
    def answer(self, qid, question, role, as_of) -> AnswerResult:
        import time
        t0 = time.perf_counter()
        if role not in ROLE_RANK:
            raise ValueError(f"unknown role {role!r}")
        # Evidence for the answer is top-6 (the reference depth); the
        # blocked-evidence signal scans to k=10 so a denied document that
        # would have ranked 7-10 still flips the refusal to not_permitted.
        # Only the boolean is used - denied text never leaves the signal.
        r = self.retr.retrieve(question, role, as_of, k=10)
        hits = r.hits[:6]
        debug = {"blocked_evidence": r.blocked_evidence, "allowed_pool": r.allowed_pool,
                 "top_cosine": r.top_cosine, "top_bm25": r.top_bm25,
                 "top_score": hits[0].score if hits else 0.0}

        def finish(res):
            res.latency_ms = (time.perf_counter() - t0) * 1000
            res.debug = debug
            if not self._role_safe(role, res.answer, res.citations):
                # Defense in depth: never emit anything the guard rejects.
                print(f"ROLE GUARD tripped on {qid}; replaced with bare refusal",
                      file=sys.stderr)
                res = AnswerResult(id=qid, status="not_permitted",
                                   answer=NOT_PERMITTED_TEXT, citations=[])
                res.latency_ms = (time.perf_counter() - t0) * 1000
                res.debug = {**debug, "role_guard": True}
            return res

        if not hits or r.allowed_pool == 0:
            status = "not_permitted" if r.blocked_evidence else "not_in_corpus"
            text = NOT_PERMITTED_TEXT if r.blocked_evidence else NOT_IN_CORPUS_TEXT
            return finish(AnswerResult(id=qid, status=status, answer=text, citations=[]))
        if r.blocked_evidence and r.top_bm25 < NP_BM25:
            return finish(AnswerResult(id=qid, status="not_permitted",
                                       answer=NOT_PERMITTED_TEXT, citations=[]))
        if r.top_bm25 < OOC_BM25 and r.top_cosine < OOC_COS:
            return finish(AnswerResult(id=qid, status="not_in_corpus",
                                       answer=NOT_IN_CORPUS_TEXT, citations=[]))

        if self.gen_backend == "openai":
            from . import llm
            try:
                out = llm.generate(question, as_of, hits, self._verify_quote,
                                   meta=self._gen_meta(as_of))
            except llm.QuotaExceeded:
                if os.environ.get("OP05_GEN_STRICT") == "1":
                    raise
                print(f"/ask {qid}: generator quota exhausted; extractive fallback",
                      file=sys.stderr)
                out = None
            if out is not None:
                text, cit_dicts, usage = out
                cost = (usage["input_tokens"] * self._price_in
                        + usage["output_tokens"] * self._price_out) / 1e6
                res = AnswerResult(
                    id=qid, status="answered", answer=text,
                    citations=[Citation(**c) for c in cit_dicts],
                    confidence=min(1.0, max(0.0, hits[0].score)),
                    cost_usd=round(cost, 8),
                    generator=f"llm:{self.gen_model or 'env'}")
                res = finish(res)
                res.debug["gen_usage"] = usage
                return res
            if out is None:
                debug["gen_fallback_reason"] = getattr(llm, "last_error", None)
            # generation unavailable/unverifiable/INSUFFICIENT -> extractive

        picked = self._best_window(question, hits)
        if picked is None:
            # Retrieval passed the thresholds but no supportable evidence
            # window exists - refuse rather than fabricate.
            return finish(AnswerResult(id=qid, status="not_in_corpus",
                                       answer=NOT_IN_CORPUS_TEXT, citations=[]))
        best, second = picked
        text, citations = self._compose(question, as_of, best, second)
        conf = min(1.0, max(0.0, hits[0].score))
        return finish(AnswerResult(id=qid, status="answered", answer=text,
                                   citations=citations, confidence=conf))
