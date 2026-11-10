"""Optional budget-class LLM answer generator. OFF by default.

The default generator is the deterministic extractive composer in
policybook.answer (zero API tokens, fully offline, clearly labeled). This
module activates when OP05_GEN_BACKEND=openai is set explicitly, or
automatically when reviewer-gateway credentials (LLM_BASE_URL +
LLM_API_KEY) are present, so reproduce.sh serves model answers with no
OP05-specific flag. LLM_MODEL is omittable: unnamed, the request omits
"model" and the gateway serves its default model.

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
  instead of silently degrading to the extractive fallback (a run measured
  half on the fallback would be mislabeled evidence). One exception
  (user-approved 2026-10-27): LLM_API_KEY_FALLBACK configures a second key
  for the SAME model and endpoint; a 429 fails over to it key-level only,
  never a model or behavior change, recorded in usage so the run is labeled.
  QuotaExceeded raises only when both keys are exhausted or no fallback is
  configured.
"""
from __future__ import annotations

import json
from contextvars import ContextVar
from contextlib import contextmanager
import os
import time
import urllib.error
import urllib.request

PROMPT = """Answer the QUESTION using ONLY the EVIDENCE passages below.

Rules:
- No outside knowledge; every fact must come from the evidence.
- Be concise but complete: every fact asked for (numbers, dates, thresholds, conditions, rule names). If the evidence says a rule is conditional, has an exception, or is superseded/withdrawn as of the as-of date, say so.
- Cite the passage(s) behind the facts. Each quote VERBATIM from the passage - exact characters, no paraphrase, no ellipsis, no merged non-adjacent sentences.
- Derived results: when the answer computes or combines facts (arithmetic, date math, threshold checks, comparisons), cite the passage(s) stating the INPUT figures or rules you used, and show the derivation briefly in the answer (e.g. "X - Y = Z"). A quote backs the facts you used - never cite a passage as if it stated the computed result.
- INSUFFICIENT is for ABSENT facts, not for work: if the evidence states the underlying rule, DO the required reasoning - arithmetic, date calculation, threshold checks, and comparing two passages or versions - and answer with it. Answer "INSUFFICIENT" (no citations) only when the facts needed are not in the evidence at all. Never guess.
- JSON only: {{"answer": "...", "citations": [{{"doc_id": "...", "section": "...", "quote": "..."}}]}} or {{"answer": "INSUFFICIENT", "citations": []}}

QUESTION (as of {as_of}):
{question}

EVIDENCE:
{evidence}"""


# EXPERIMENTAL (2026-11-09, DECISIONS 31): date-asking clause, OFF by default.
# Enable with OP05_GEN_DATE_CLAUSE=1. Isolated after the v3 dev run showed the
# clause fixes its target class (date-asking temporal rows) without breaking
# as-of version controls but did not show controlled net benefit over the
# baseline prompt on the full dev judged pass. Default prompt = v2 text.
DATE_CLAUSE = """- Date-asking questions: when the question asks WHEN a rule, framework or requirement
  was introduced, inserted, amended or takes effect, report the dated facts in the
  evidence as dated facts - including a date later than the as-of date - and cite the
  passage stating them. The as-of date still selects which version of a rule governs
  for what-is-the-rule questions; it does not make a stated introduction or amendment
  date unsayable.
"""


def _active_prompt():
    if os.environ.get("OP05_GEN_DATE_CLAUSE") == "1":
        return PROMPT.replace("- JSON only:", DATE_CLAUSE + "- JSON only:")
    return PROMPT


class QuotaExceeded(Exception):
    """The provider refused the call on rate/quota grounds (HTTP 429)."""


last_error = None  # legacy single-thread debug view; never use for decisions
_error = ContextVar("generation_error", default=None)
_meter = ContextVar("generation_meter", default=None)


def _set_error(value):
    global last_error
    last_error = value
    _error.set(value)


def get_last_error():
    return _error.get()


@contextmanager
def request_context():
    """Isolate request state and meter every HTTP attempt, including failures.

    No prompts, response text, provider details or credentials enter this meter.
    Provider errors may omit usage, so unknown usage is counted, never zeroed.
    """
    meter = {"attempts": 0, "input_tokens": 0, "output_tokens": 0,
             "usage_unknown_attempts": 0, "errors": {}}
    mt = _meter.set(meter)
    et = _error.set(None)
    try:
        yield meter
    finally:
        _error.reset(et)
        _meter.reset(mt)


def _post(base_url, api_key, payload, timeout=90):
    meter = _meter.get()
    if meter is not None:
        meter["attempts"] += 1
    try:
        data = _post_impl(base_url, api_key, payload, timeout)
    except Exception as exc:
        if meter is not None:
            meter["usage_unknown_attempts"] += 1
            category = ("quota" if isinstance(exc, QuotaExceeded) else
                        "transient" if isinstance(exc, TransientUnavailable) else
                        "other")
            meter["errors"][category] = meter["errors"].get(category, 0) + 1
        raise
    if meter is not None:
        u = data.get("usage") or {}
        if "prompt_tokens" not in u or "completion_tokens" not in u:
            meter["usage_unknown_attempts"] += 1
        meter["input_tokens"] += u.get("prompt_tokens", 0)
        meter["output_tokens"] += u.get("completion_tokens", 0)
    return data


def _post_impl(base_url, api_key, payload, timeout=90):
    """One HTTP attempt under a single wall-clock budget: `timeout` covers
    connect + request + response headers + response body together, not each
    phase separately. The connect phase keeps the socket timeout as its own
    cap; whatever budget it leaves is the body's deadline."""
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise TransientUnavailable(
                    f"headers consumed the whole request budget ({timeout}s)")
            return json.loads(_read_bounded(r, deadline=remaining))
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




def _read_bounded(resp, deadline):
    """Read a response body under a TOTAL wall-clock deadline.

    urllib's socket timeout caps each blocking operation, not the whole
    transfer: a stalled endpoint that trickles a byte every few seconds
    holds a read open indefinitely (observed live 2026-10-20: single
    requests blocking 10+ minutes while still returning 200s). read1
    returns after each underlying recv, so a slow trickle trips the
    deadline and becomes a bounded TransientUnavailable that the 503
    backoff can retry instead of hanging the run.
    """
    end = time.monotonic() + deadline
    buf = bytearray()
    while True:
        remaining = end - time.monotonic()
        if remaining <= 0:
            raise TransientUnavailable(
                f"response body exceeded total deadline ({deadline}s)")
        try:
            sock = resp.fp.raw._sock
            sock.settimeout(min(30.0, remaining))
        except Exception:  # noqa: BLE001 - non-socket stub responses
            pass
        try:
            chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)
        except (TimeoutError, OSError) as e:
            raise TransientUnavailable(f"stalled response body: {e}") from e
        if not chunk:
            return bytes(buf)
        buf.extend(chunk)

class TransientUnavailable(Exception):
    """HTTP 503 high-demand - temporary provider capacity, not quota."""


def _post_with_503_backoff(base_url, api_key, payload, timeout=90):
    """503 'high demand' is transient capacity, not quota: a few short
    backoff retries are honest recovery. 429 quota still stops the run.

    The whole call - every attempt's connect/request/headers/body plus the
    backoff sleeps - is bounded by one total deadline of 4*timeout + 30s
    (worst case ~6.5 min at the default), so no code path can hang the run
    indefinitely."""
    total_deadline = time.monotonic() + 4 * timeout + 30
    for attempt in range(4):
        remaining = total_deadline - time.monotonic()
        if remaining <= 0:
            raise TransientUnavailable(
                f"total request deadline exhausted after {attempt} attempts")
        try:
            return _post(base_url, api_key, payload, min(timeout, remaining))
        except TransientUnavailable:
            if attempt == 3:
                raise
            time.sleep(min(5 * (attempt + 1),
                           max(0.0, total_deadline - time.monotonic())))


def _chat(base_url, api_key, model, messages, timeout=90, one_attempt=False):
    """One chat completion, adapting the request shape to provider dialect.

    First attempt mirrors the grader's judge client (no temperature,
    max_completion_tokens). A 400 that names max_completion_tokens retries
    once with max_tokens; a 400 that names response_format retries once
    without JSON mode. Adaptations are per-call and recorded.

    model may be empty: the "model" field is then omitted and the gateway
    serves its default model. one_attempt=True (the live service path)
    skips the 503 backoff loop: exactly one bounded HTTP attempt, and a
    TransientUnavailable propagates so the caller can answer 503."""
    json_mode = os.environ.get("OP05_GEN_JSON_MODE", "1") != "0"
    payload = {"max_completion_tokens": 2500, "messages": messages}
    if model:
        payload["model"] = model
    # Reasoning models burn output tokens on hidden thinking; the challenge
    # output budget is measured on usage, so ask for low effort when the
    # provider supports it (adapted away on a 400).
    payload["reasoning_effort"] = os.environ.get("OP05_GEN_REASONING", "low")
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    adaptations = []
    for _ in range(3):
        try:
            post = _post if one_attempt else _post_with_503_backoff
            data = post(base_url, api_key, payload, timeout)
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
             base_url=None, api_key=None, meta=None, one_attempt=False):
    """hits: role-filtered retrieval hits; verify_quote(doc_id, section, quote)->bool;
    meta: optional {doc_id: {title, effective_date, later_version, later_date}}.

    Returns (answer_text, citations, usage) or None when generation failed /
    was INSUFFICIENT / nothing verifiable remained (caller then falls back to
    the extractive generator). Raises QuotaExceeded on provider quota signals
    after the optional key-level failover (LLM_API_KEY_FALLBACK, second key
    for the same model/endpoint) is exhausted or absent.
    Model resolution: explicit arg, then OP05_GEN_MODEL, then LLM_MODEL; when
    all are empty the request omits "model" and the gateway default is used.
    one_attempt=True (live path): a single bounded HTTP attempt, and
    TransientUnavailable (timeout/stall/provider 503) raises instead of
    degrading into a silent extractive fallback.
    """
    global last_error
    _set_error(None)
    model = (model or os.environ.get("OP05_GEN_MODEL")
             or os.environ.get("LLM_MODEL", ""))
    base_url = base_url or os.environ.get("LLM_BASE_URL", "")
    api_key = api_key if api_key is not None else os.environ.get("LLM_API_KEY", "")
    fallback_key = os.environ.get("LLM_API_KEY_FALLBACK", "")
    if not base_url:
        _set_error("not configured")
        return None
    evidence = "\n\n".join(pack_evidence(hits, meta))
    messages = [{"role": "user",
                 "content": _active_prompt().format(as_of=as_of, question=question,
                                          evidence=evidence)}]
    usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0,
             "adaptations": []}
    for attempt in (1, 2):
        try:
            data, adaptations = _chat(base_url, api_key, model, messages,
                                      one_attempt=one_attempt)
        except QuotaExceeded:
            # Key-level failover: the second key serves the SAME model and
            # endpoint, so a converted call is identical evidence - only the
            # quota project changes. The switch is recorded in usage and
            # persists for the rest of the run. With no fallback configured,
            # or when the fallback also 429s, the quota signal stays loud.
            if fallback_key and api_key != fallback_key:
                api_key = fallback_key
                usage["key_fallback_used"] = True
                data, adaptations = _chat(base_url, api_key, model, messages,
                                          one_attempt=one_attempt)
            else:
                raise  # quota signals must stay loud; caller decides
        except TransientUnavailable:
            if one_attempt:
                raise  # live path: temporary error, never silent fallback
            _set_error("temporarily unavailable")
            return None
        except Exception as e:  # noqa: BLE001 - a flaky API must not end the run
            _set_error(f"{type(e).__name__}: {e}"[:300])
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
            _set_error("unparseable response")
            return None
        answer = obj.get("answer", "")
        if not isinstance(answer, str) or not answer.strip():
            _set_error("empty answer")
            return None
        if answer.strip().upper() == "INSUFFICIENT":
            _set_error("model said INSUFFICIENT")
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
                _set_error("no verifiable citations after retry")
                return None
            return answer, cits, usage
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content":
                         f"{dropped} citation quote(s) were not verbatim substrings of the "
                         "cited passage. Copy quotes exactly and reply again as JSON."})
    return None
