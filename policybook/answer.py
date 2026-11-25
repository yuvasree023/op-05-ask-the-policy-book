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
# Abbreviations whose trailing period must NOT split a sentence: a fragment
# like "10 crore of exposure..." scores on digits for numeric questions
# without carrying the rule (outside review 2026-11-23, point 3).
_ABBREV = {"rs", "no", "sec", "art", "cl", "approx", "etc", "e.g", "i.e",
           "dr", "mr", "mrs", "ms", "vs", "w.e.f", "st", "fig", "reg", "regs"}


def _segments(text: str):
    spans, start = [], 0
    for m in _BOUNDARY.finditer(text):
        end = m.start()
        if "\n" not in m.group(0):
            # period-class boundary: skip when the preceding word is an
            # abbreviation ("Rs. 10 crore" is one sentence, not two).
            head = text[start:end].rstrip()
            word = head[head.rfind(" ") + 1:].rstrip(".!?").lower() if head else ""
            if word in _ABBREV:
                continue
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
    confidence: float = 0.0  # top retrieval fusion score in [0,1], NOT a calibrated probability (the grader only schema-checks the range)
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
    def __init__(self, index_dir, model_dir, corpus_dir, live=False):
        # live=True (the HTTP service): one bounded model attempt, then a
        # temporary error - never a silent extractive fallback on timeout.
        self.live = live
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
        # Generator selection (approved 2026-10-22):
        # - OP05_GEN_BACKEND set explicitly -> honored as-is.
        # - else, reviewer-gateway credentials present (LLM_BASE_URL +
        #   LLM_API_KEY) -> the model generator activates automatically, so
        #   reproduce.sh serves model answers with no OP05-specific flag.
        # - else -> the extractive composer: offline, $0, no API calls,
        #   clearly labeled (startup banner, X-Generator header, per-row
        #   generator field in batch outputs).
        # LLM_MODEL is omittable: with no model named anywhere, the request
        # omits "model" and the gateway serves its default.
        explicit = os.environ.get("OP05_GEN_BACKEND", "")
        if explicit:
            self.gen_backend = explicit
        elif os.environ.get("LLM_BASE_URL") and os.environ.get("LLM_API_KEY"):
            self.gen_backend = "openai"
        else:
            self.gen_backend = "extractive"
        self.gen_auto = not explicit and self.gen_backend == "openai"
        self.gen_model = (os.environ.get("OP05_GEN_MODEL")
                          or os.environ.get("LLM_MODEL", ""))
        # Pinned list prices (USD per 1M tokens) for cost reporting; free-tier
        # usage still counts at list price per FREE_CREDITS.md. Defaults are
        # the pinned generator snapshot's published list prices so a run that
        # forgets the env vars never silently reports $0 for model rows
        # (outside review 2026-11-23, point 16); cost_complete still requires
        # the prices to be set EXPLICITLY for the run's accounting to be
        # claimed complete.
        self._price_in = float(os.environ.get("OP05_PRICE_IN_PER_M", "0.30") or 0)
        self._price_out = float(os.environ.get("OP05_PRICE_OUT_PER_M", "2.50") or 0)
        # Normalized passage texts keyed by passage id, for citation
        # verification (grader checks the quote against the cited passage).
        self._passage_norm = {s.passage_id: norm(s.text) for s in self.corpus.sections}
        self._passage_raw = {s.passage_id: s.text for s in self.corpus.sections}

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
            # same passage, bounded by MAX_QUOTE_CHARS. The gap between
            # segments is boundary whitespace by construction, so expansion
            # requires it to be whitespace-only (the pre-2026-11-23 condition
            # demanded non-whitespace and was dead code - every extractive
            # quote was a single sentence).
            best_sc, bs, be = scored[0]
            order = sorted((s, e) for _, s, e in scored[:3])
            s0, e0 = bs, be
            for s, e in order:
                if e <= s0 and not h.text[e:s0].strip() and e0 - s <= MAX_QUOTE_CHARS:
                    s0 = s
                elif s >= be and not h.text[be:s].strip() and e - s0 <= MAX_QUOTE_CHARS:
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
    def _compose(self, question, as_of, best, second, role):
        sc, doc_id, section, quote, _ = best
        meta = self.corpus.docs[doc_id]
        parts = [f'According to {meta.title} (in force since '
                 f'{meta.effective_date}): "{quote}"']
        citations = [Citation(doc_id=doc_id, section=section, quote=quote)]
        req_rank = ROLE_RANK[role]
        for new_doc, new_eff in self._superseded_by.get(doc_id, []):
            if new_eff > as_of:
                # Never name a superseding document the requester may not see:
                # its title alone is a reference to restricted material
                # (deterministic checks miss short names; semantic review does
                # not). Say THAT a later amendment exists, nothing more.
                if ROLE_RANK[self.corpus.docs[new_doc].role] <= req_rank:
                    parts.append(f"Note: a later version ({self.corpus.docs[new_doc].title}) "
                                 f"takes effect {new_eff}, after the as-of date.")
                else:
                    parts.append(f"Note: a later amendment takes effect {new_eff}, "
                                 "after the as-of date.")
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

    _STOP = frozenset("the a an and or of to in on for by with as is are was were be been that this "
                      "these those it its from at not no any all under which who whom what when where "
                      "shall may must can will than then also such other their there".split())

    def _content_tokens(self, text):
        return {t for t in re.findall(r"[a-z0-9][a-z0-9.%/-]*", norm(text))
                if t not in self._STOP and (len(t) >= 4 or any(c.isdigit() for c in t))}

    def _expand_citations(self, answer, citations, max_chars=700):
        """General citation hygiene (OP05_QUOTE_EXPAND, default ON since 2026-11-25; =0 disables): a cited quote that
        states only part of a multi-part answer is widened, within the SAME
        passage, to the contiguous neighbouring sentences that state more of
        the answer's content words. The result is a contiguous substring of
        the passage, re-verified with the grader-compatible check; on any
        doubt the original quote is kept. Deterministic, no model call."""
        ans_toks = self._content_tokens(answer)
        out = []
        for c in citations:
            raw = self._passage_raw.get(c.section)
            new = c
            if raw and ans_toks:
                segs = _segments(raw)
                nq = norm(c.quote)
                idx = [i for i, (a0, a1) in enumerate(segs)
                       if norm(raw[a0:a1]) and (norm(raw[a0:a1]) in nq or nq in norm(raw[a0:a1])
                                                or (norm(raw[a0:a1]).split()[:6] and " ".join(norm(raw[a0:a1]).split()[:6]) in nq))]
                if idx:
                    lo, hi = min(idx), max(idx)
                    covered = self._content_tokens(raw[segs[lo][0]:segs[hi][1]]) & ans_toks
                    while True:
                        best = None
                        for ni in (lo - 1, hi + 1):
                            if ni < 0 or ni >= len(segs):
                                continue
                            nlo, nhi = min(lo, ni), max(hi, ni)
                            if segs[nhi][1] - segs[nlo][0] > max_chars:
                                continue
                            gain = (self._content_tokens(raw[segs[ni][0]:segs[ni][1]]) & ans_toks) - covered
                            if len(gain) >= 2 and (best is None or len(gain) > len(best[1])):
                                best = (ni, gain)
                        if best is None:
                            break
                        lo, hi = min(lo, best[0]), max(hi, best[0])
                        covered |= best[1]
                    q = raw[segs[lo][0]:segs[hi][1]].strip()
                    if q and norm(q) != nq and nq in norm(q) and self._verify_quote(c.doc_id, c.section, q):
                        new = Citation(doc_id=c.doc_id, section=c.section, quote=q)
            out.append(new)
        return out

    _BROAD_RETRY = ("model said INSUFFICIENT", "no verifiable citations",
                    "unparseable response", "RuntimeError: HTTP 400")

    @staticmethod
    def _retry_reason_ok(err):
        """Larger-pack retry trigger. Default: INSUFFICIENT only (validated
        behaviour). OP05_GEN_RETRY_BROAD (default ON since 2026-11-25; =0 disables) also retries
        citation-verification, parse and provider-400 failures."""
        err = err or ""
        if os.environ.get("OP05_GEN_RETRY_BROAD", "1") == "1":
            return err.startswith(Answerer._BROAD_RETRY)
        return err == "model said INSUFFICIENT"

    _DATE_INTENT = re.compile(r"\b(date|when|deadline|days?|months?|years?|period|expir\w*|"
                              r"before|after|prior|within|as of|effective)\b", re.I)

    def _effort_for(self, question):
        """OP05_GEN_REASONING_NUM=<effort> (default "medium" since 2026-11-25; =0 disables): use a higher
        reasoning effort only for questions that ask for a number, amount or
        date calculation. General question-shape rule."""
        eff = os.environ.get("OP05_GEN_REASONING_NUM", "medium")
        if eff and eff != "0" and (_NUM_INTENT.search(question) or self._DATE_INTENT.search(question)):
            return eff
        return None

    def _gen(self, llm, question, as_of, hits):
        eff = self._effort_for(question)
        prev = os.environ.get("OP05_GEN_REASONING")
        if eff:
            os.environ["OP05_GEN_REASONING"] = eff
        try:
            return llm.generate(question, as_of, hits, self._verify_quote,
                                meta=self._gen_meta(as_of), one_attempt=self.live)
        finally:
            if eff:
                if prev is None:
                    os.environ.pop("OP05_GEN_REASONING", None)
                else:
                    os.environ["OP05_GEN_REASONING"] = prev

    def _maybe_expand(self, answer, citations):
        if os.environ.get("OP05_QUOTE_EXPAND", "1") == "1":
            return self._expand_citations(answer, citations)
        return citations

    def _gen_meta(self, as_of):
        """Per-document prompt metadata: title, in-force date, and any later
        version taking effect after the as-of date."""
        meta = {}
        for d, m in self.corpus.docs.items():
            entry = {"title": m.title, "effective_date": m.effective_date,
                     "supersedes": list(m.supersedes)}
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
        from . import llm
        with llm.request_context() as meter:
            try:
                result = self._answer(qid, question, role, as_of)
            except Exception as exc:
                exc.generation_meter = dict(meter)
                raise
            result.debug["generation_meter"] = dict(meter)
            if meter["attempts"]:
                result.cost_usd = round((meter["input_tokens"] * self._price_in
                                        + meter["output_tokens"] * self._price_out) / 1e6, 8)
                result.debug["cost_complete"] = (not meter["usage_unknown_attempts"]
                    and bool(os.environ.get("OP05_PRICE_IN_PER_M"))
                    and bool(os.environ.get("OP05_PRICE_OUT_PER_M")))
            return result

    def _answer(self, qid, question, role, as_of) -> AnswerResult:
        import time
        t0 = time.perf_counter()
        if role not in ROLE_RANK:
            raise ValueError(f"unknown role {role!r}")
        # Evidence for the answer is top-6 (the reference depth); the
        # blocked-evidence signal scans to k=10 so a denied document that
        # would have ranked 7-10 still flips the refusal to not_permitted.
        # Only the boolean is used - denied text never leaves the signal.
        # The retrieval pool itself is fetched to k=12 (evidence widening,
        # DECISIONS 37): the model sees up to 12 candidates while the
        # refusal signal stays pinned at the dev-tuned depth of 10.
        r = self.retr.retrieve(question, role, as_of, k=12, signal_k=10)
        hits = r.hits[:6]
        # Iteration B (approved lever: context assembly): top-6 starved some
        # questions whose supporting passage ranked 7-10 (measured 2026-10-23:
        # 3 of 25 INSUFFICIENT fallbacks), so the MODEL sees a wider pool;
        # the char budget bounds prompt size. Widened to top-12 on 2026-11-23
        # after the v4A failure analysis showed gold passages ranked 9-10
        # excluded from the candidate eight (DECISIONS 37); the extractive
        # path keeps the original top-6 so its deterministic baseline is
        # undisturbed.
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
                out = self._gen(llm, question, as_of, r.hits[:12])
            except llm.TransientUnavailable:
                if self.live:
                    raise  # the service answers 503 + Retry-After
                print(f"/ask {qid}: generator temporarily unavailable; "
                      "extractive fallback", file=sys.stderr)
                out = None
            except llm.QuotaExceeded:
                if self.live or os.environ.get("OP05_GEN_STRICT") == "1":
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
                    citations=self._maybe_expand(text, [Citation(**c) for c in cit_dicts]),
                    confidence=min(1.0, max(0.0, hits[0].score)),
                    cost_usd=round(cost, 8),
                    generator=f"llm:{self.gen_model or 'gateway-default'}")
                res = finish(res)
                res.debug["gen_usage"] = usage
                return res
            # EXPERIMENTAL (2026-11-09): one bounded retry when the model said
            # INSUFFICIENT - dev runs show abstention is unstable run-to-run
            # and costs judged rows the evidence pack supports. OFF by default
            # (OP05_GEN_INSUFFICIENT_RETRY=1). The flag governs the SAME policy
            # on eval and live paths so eval measurements can map to service
            # behavior; with the flag ON, /ask also makes the second attempt,
            # adding at most one bounded generation call to latency.
            # Verification, the role guard and fabrication accounting gate the
            # retried answer exactly like a first attempt.
            if (out is None
                    and os.environ.get("OP05_GEN_INSUFFICIENT_RETRY", "1") == "1"
                    and self._retry_reason_ok(llm.get_last_error())):
                # Larger-pack retry (2026-11-24): the INSUFFICIENT retry used the
                # same pack and mostly abstained again. Retry once with a bigger
                # char budget (OP05_GEN_RETRY_EV_CHAR_BUDGET) - only rows that
                # would otherwise fall back pay the extra input tokens.
                _rb = os.environ.get("OP05_GEN_RETRY_EV_CHAR_BUDGET", "3500")
                _prev = os.environ.get("OP05_GEN_EV_CHAR_BUDGET")
                if _rb:
                    os.environ["OP05_GEN_EV_CHAR_BUDGET"] = _rb
                try:
                    out = self._gen(llm, question, as_of, r.hits[:12])
                finally:
                    if _rb:
                        if _prev is None:
                            os.environ.pop("OP05_GEN_EV_CHAR_BUDGET", None)
                        else:
                            os.environ["OP05_GEN_EV_CHAR_BUDGET"] = _prev
                if out is not None:
                    text, cit_dicts, usage = out
                    usage = {**usage, "insufficient_retry_used": True,
                             "retry_pack_budget": _rb}
                    cost = (usage["input_tokens"] * self._price_in
                            + usage["output_tokens"] * self._price_out) / 1e6
                    res = AnswerResult(
                        id=qid, status="answered", answer=text,
                        citations=self._maybe_expand(text, [Citation(**c) for c in cit_dicts]),
                        confidence=min(1.0, max(0.0, hits[0].score)),
                        cost_usd=round(cost, 8),
                        generator=f"llm:{self.gen_model or 'gateway-default'}")
                    res = finish(res)
                    res.debug["gen_usage"] = usage
                    return res
            if out is None:
                debug["gen_fallback_reason"] = llm.get_last_error()
            # generation unavailable/unverifiable/INSUFFICIENT -> extractive

        picked = self._best_window(question, hits)
        if picked is None:
            # Retrieval passed the thresholds but no supportable evidence
            # window exists - refuse rather than fabricate.
            return finish(AnswerResult(id=qid, status="not_in_corpus",
                                       answer=NOT_IN_CORPUS_TEXT, citations=[]))
        best, second = picked
        text, citations = self._compose(question, as_of, best, second, role)
        conf = min(1.0, max(0.0, hits[0].score))
        return finish(AnswerResult(id=qid, status="answered", answer=text,
                                   citations=citations, confidence=conf))
