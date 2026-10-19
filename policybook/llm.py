"""Optional budget-class LLM answer generator. OFF by default.

The default generator is the deterministic extractive composer in
policybook.answer (zero API tokens, fully offline). This module is used only
when OP05_GEN_BACKEND=openai is set explicitly; reproduce.sh does not set it,
so the offline/$0 default is unchanged.

Design constraints carried from the challenge contract:
- budget-class models only (MODELS.md: mini/nano/Flash/Haiku tier or
  open-weight <= ~35B); the exact snapshot must be named and disclosed, with
  a public price source and rates recorded in the run manifest (free-tier
  usage still counts at pinned list price per FREE_CREDITS.md);
- only role/temporal-eligible evidence enters the prompt - above-role text
  is never sent anywhere, including to an external API;
- every returned citation quote is verified as a normalized substring of the
  cited passage before use; unverifiable citations trigger one retry, then
  are dropped; if nothing verifiable remains the caller falls back to the
  extractive generator;
- provider differences are absorbed adaptively (max_completion_tokens vs
  max_tokens, optional JSON mode) instead of pinning one vendor's dialect;
- a 429/quota signal raises QuotaExceeded so evaluation runs STOP loudly
  instead of silently degrading to the fallback (a run measured half on the
  fallback would be mislabeled evidence).
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

PROMPT = """Answer the QUESTION using ONLY the EVIDENCE passages below.

Rules:
- No outside knowledge; every fact must come from the evidence.
- Be concise but complete: every fact asked for (numbers, dates, thresholds, conditions, rule names). If the evidence says a rule is conditional, has an exception, or is superseded/withdrawn as of the as-of date, say so.
- Cite the passage(s) behind the facts. Each quote VERBATIM from the passage - exact characters, no paraphrase, no ellipsis, no merged non-adjacent sentences.
- If the evidence lacks the answer: "INSUFFICIENT", no citations. Never guess.
- JSON only: {{"answer": "...", "citations": [{{"doc_id": "...", "section": "...", "quote": "..."}}]}} or {{"answer": "INSUFFICIENT", "citations": []}}

QUESTION (as of {as_of}):
{question}

EVIDENCE:
{evidence}"""


class QuotaExceeded(Exception):
    """The provider refused the call on rate/quota grounds (HTTP 429)."""


last_error = None  # why the last generate() returned None (debug visibility)


def _post(base_url, api_key, payload, timeout=90):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode()[:500]
        except Exception:  # noqa: BLE001
            pass
        if e.code == 429:
            raise QuotaExceeded(f"429 from {base_url}: {detail}") from e
        if e.code == 503:
            raise TransientUnavailable(detail) from e
        raise RuntimeError(f"HTTP {e.code} from {base_url}: {detail}") from e


class TransientUnavailable(Exception):
    """HTTP 503 high-demand - temporary provider capacity, not quota."""


def _post_with_503_backoff(base_url, api_key, payload, timeout=90):
    """503 'high demand' is transient capacity, not quota: a few short
    backoff retries are honest recovery. 429 quota still stops the run."""
    for attempt in range(4):
        try:
            return _post(base_url, api_key, payload, timeout)
        except TransientUnavailable:
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))


def _chat(base_url, api_key, model, messages, timeout=90):
    """One chat completion, adapting the request shape to provider dialect.

    First attempt mirrors the grader's judge client (no temperature,
    max_completion_tokens). A 400 that names max_completion_tokens retries
    once with max_tokens; a 400 that names response_format retries once
    without JSON mode. Adaptations are per-call and recorded.
    """
    json_mode = os.environ.get("OP05_GEN_JSON_MODE", "1") != "0"
    payload = {"model": model, "max_completion_tokens": 2500,
               "messages": messages}
    # Reasoning models burn output tokens on hidden thinking; the challenge
    # output budget is measured on usage, so ask for low effort when the
    # provider supports it (adapted away on a 400).
    payload["reasoning_effort"] = os.environ.get("OP05_GEN_REASONING", "low")
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    adaptations = []
    for _ in range(3):
        try:
            data = _post_with_503_backoff(base_url, api_key, payload, timeout)
            return data, adaptations
        except RuntimeError as e:
            msg = str(e).lower()
            if "max_completion_tokens" in msg and "max_completion_tokens" in payload:
                payload["max_tokens"] = payload.pop("max_completion_tokens")
                adaptations.append("max_tokens")
                continue
            if "reasoning_effort" in msg and "reasoning_effort" in payload:
                payload.pop("reasoning_effort")
                adaptations.append("no_reasoning_param")
                continue
            if "response_format" in msg and "response_format" in payload:
                payload.pop("response_format")
                adaptations.append("no_json_mode")
                continue
            raise
    raise RuntimeError("request shape adaptation exhausted")


def _evidence_block(h, meta):
    """Compact header: identity + temporal markers only (long titles cost
    tokens and the citation contract uses doc_id/section, not titles)."""
    m = (meta or {}).get(h.doc_id, {})
    head = f"[{h.doc_id} | {h.section}"
    if m.get("effective_date"):
        head += f" | in force {m['effective_date']}"
        if m.get("later_version"):
            head += f"; later version in force {m.get('later_date')}"
    head += "]"
    return f"{head}\n{h.text}"


def pack_evidence(hits, meta, char_budget=None, min_passages=3):
    """Budget-fill: include ranked passages whole until the character budget
    is exhausted (cost bar 8 caps input tokens/question; dropping the weakest
    passage beats truncating a passage mid-rule). Always keeps at least
    min_passages when available. Budget from OP05_GEN_EV_CHAR_BUDGET."""
    if char_budget is None:
        char_budget = int(os.environ.get("OP05_GEN_EV_CHAR_BUDGET", "2600"))
    blocks, used = [], 0
    for i, h in enumerate(hits):
        b = _evidence_block(h, meta)
        if blocks and used + len(b) > char_budget and len(blocks) >= min_passages:
            break
        blocks.append(b)
        used += len(b)
    return blocks


def generate(question, as_of, hits, verify_quote, model=None,
             base_url=None, api_key=None, meta=None):
    """hits: role-filtered retrieval hits; verify_quote(doc_id, section, quote)->bool;
    meta: optional {doc_id: {title, effective_date, later_version, later_date}}.

    Returns (answer_text, citations, usage) or None when generation failed /
    was INSUFFICIENT / nothing verifiable remained (caller then falls back to
    the extractive generator). Raises QuotaExceeded on provider quota signals.
    """
    global last_error
    last_error = None
    model = model or os.environ.get("OP05_GEN_MODEL", "")
    base_url = base_url or os.environ.get("LLM_BASE_URL", "")
    api_key = api_key if api_key is not None else os.environ.get("LLM_API_KEY", "")
    if not (model and base_url):
        last_error = "not configured"
        return None
    evidence = "\n\n".join(pack_evidence(hits, meta))
    messages = [{"role": "user",
                 "content": PROMPT.format(as_of=as_of, question=question,
                                          evidence=evidence)}]
    usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0,
             "adaptations": []}
    for attempt in (1, 2):
        try:
            data, adaptations = _chat(base_url, api_key, model, messages)
        except QuotaExceeded:
            raise  # quota signals must stay loud; caller decides
        except Exception as e:  # noqa: BLE001 - a flaky API must not end the run
            last_error = f"{type(e).__name__}: {e}"[:300]
            return None
        usage["calls"] += 1
        usage["adaptations"] = sorted(set(usage["adaptations"]) | set(adaptations))
        u = data.get("usage") or {}
        usage["input_tokens"] += u.get("prompt_tokens", 0)
        usage["output_tokens"] += u.get("completion_tokens", 0)
        try:
            text = data["choices"][0]["message"]["content"] or ""
            obj = json.loads(text[text.index("{"):text.rindex("}") + 1])
        except (KeyError, IndexError, ValueError):
            last_error = "unparseable response"
            return None
        answer = obj.get("answer", "")
        if not isinstance(answer, str) or not answer.strip():
            last_error = "empty answer"
            return None
        if answer.strip().upper() == "INSUFFICIENT":
            last_error = "model said INSUFFICIENT"
            return None
        cits, dropped = [], 0
        for c in obj.get("citations") or []:
            if (isinstance(c, dict)
                    and all(isinstance(c.get(k), str) for k in ("doc_id", "section", "quote"))
                    and verify_quote(c["doc_id"], c["section"], c["quote"])):
                cits.append({"doc_id": c["doc_id"], "section": c["section"],
                             "quote": c["quote"]})
            else:
                dropped += 1
        if cits or attempt == 2:
            if not cits:
                last_error = "no verifiable citations after retry"
                return None
            return answer, cits, usage
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content":
                         f"{dropped} citation quote(s) were not verbatim substrings of the "
                         "cited passage. Copy quotes exactly and reply again as JSON."})
    return None
