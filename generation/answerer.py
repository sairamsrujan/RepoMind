"""Generate a grounded answer.

Answers come from the local Ollama chat model by default. Setting
``GENERATION_PROVIDER=api`` (plus a key) routes generation to a hosted
OpenAI-compatible model instead — and *any* failure there falls back to Ollama,
so the app keeps answering with no network. ``AnswerResult.model`` always
records the model that actually produced the text.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import requests

import config
from generation.prompt import build_messages

# A citation is "[<source_type>_<id>]" — the shape process/chunker.py assigns.
# Requiring the prefix fixes two real defects found by the end-to-end test:
#
#   1. FALSE POSITIVES. The old pattern `\[([A-Za-z0-9_#\-]+)\]` matched any
#      bracketed word, so a model quoting a commit message as
#      'Fix[es] null pointer crash' had "[es]" read as a citation and reported
#      as FABRICATED. Editorial brackets — [es], [sic], [emphasis added] — are
#      normal in quoted prose and must not look like citations.
#
#   2. SILENTLY DROPPED CITATIONS. '.' was absent from the character class, so
#      "[release_v1.2.0]" matched nothing at all. Real release citations were
#      never counted, making release-grounded answers look uncited and
#      understating citation recall.
_CITATION_TYPES = ("commit", "pr", "issue", "review", "release")
_CITATION_RE = re.compile(
    r"\[((?:" + "|".join(_CITATION_TYPES) + r")_[A-Za-z0-9_.#\-]+)\]")
_log = logging.getLogger(__name__)


def _retry_after_seconds(resp) -> float | None:
    """Parse a Retry-After header (seconds form) if the provider sent one."""
    raw = (resp.headers or {}).get("Retry-After")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


def _is_quota_or_rate_error(exc: Exception) -> bool:
    """Whether a failure is a quota/rate rejection rather than an outage.

    Worth distinguishing because the two want opposite responses: a 429 is
    returned in milliseconds, so switching to another model is nearly free,
    while a timeout or connection error has already cost the user seconds and
    another cloud attempt would cost more.
    """
    text = str(exc).lower()
    if "timeout" in text or "timed out" in text or "connection" in text:
        return False
    return ("429" in text or "rate limit" in text or "quota" in text
            or "tpd" in text or "per day" in text)


def _is_quota_exhausted(resp) -> bool:
    """True when the provider says the wait is longer than we should ever wait.

    A per-minute burst limit clears in seconds and is worth retrying. A daily
    token quota clears in *minutes to hours* — sitting on it just makes every
    query slow before falling back anyway. Detect that case and fail over to
    the local model immediately instead.
    """
    wait = _retry_after_seconds(resp)
    if wait is not None and wait > config.GENERATION_API_MAX_WAIT:
        return True
    body = (getattr(resp, "text", "") or "").lower()
    return "per day" in body or "tpd" in body or "quota" in body


@dataclass
class AnswerResult:
    text: str
    cited_chunk_ids: list[str] = field(default_factory=list)
    model: str = ""
    # Set when cloud generation was attempted but the local model answered.
    fell_back: bool = False
    fallback_reason: str = ""


def _messages_to_prompt(messages: list[dict[str, str]]) -> str:
    """Flatten chat messages into the single prompt ``chat_chain`` accepts.

    The system message must survive the flattening — it carries the citation
    format, the coverage-window rule and the prompt-injection defence — so it is
    prepended rather than dropped.
    """
    return "\n\n".join(m.get("content", "") for m in messages if m.get("content"))


# Local models sometimes emit CJK bracket forms instead of ASCII ones —
# qwen2.5, the offline fallback, writes 【pr_123】. The citation pattern is
# ASCII-only, so every citation in such an answer was invisible to both guard
# stages, and the answer was then displayed as verified with nothing backing it.
# Normalising costs nothing and removes a whole class of silent failure.
_BRACKETS = str.maketrans({
    "【": "[", "】": "]",      # CJK lenticular  (qwen2.5 offline fallback)
    "［": "[", "］": "]",      # fullwidth
    "〔": "[", "〕": "]",      # tortoise shell
})


def normalise_citation_brackets(text: str) -> str:
    """Fold non-ASCII bracket forms so citations survive extraction."""
    return (text or "").translate(_BRACKETS)


def extract_citations(text: str) -> list[str]:
    """Return the ordered, deduped chunk_ids cited inline in ``text``."""
    seen: set[str] = set()
    out: list[str] = []
    for m in _CITATION_RE.finditer(normalise_citation_brackets(text)):
        cid = m.group(1)
        if cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


class GenerationError(RuntimeError):
    pass


class Answerer:
    """Grounded answering via local Ollama, or a hosted model with fallback.

    ``use_chain`` selects between the two very different failure priorities:

    * **Interactive (default, False)** — when cloud generation fails, drop
      straight to the local model. Each additional cloud attempt costs a full
      timeout, and HANDOFF.md §3.4 records that stacking provider waits added
      ~120s to every query before falling back to a model that answers in ~10s.
      A demo must stay responsive.

    * **Offline evaluation (True)** — latency does not matter, but *provenance*
      does. A free tier's daily token cap is exhausted after a few dozen
      questions, and dropping straight to local means most of a long run is
      answered by a 7B model while the report claims a 70B one. Walking the rest
      of ``GENERATION_CHAIN`` first keeps the run on comparable cloud models.
    """

    def __init__(self, model: str | None = None, host: str | None = None,
                 use_chain: bool = False):
        self.model = model or config.GENERATION_MODEL
        self.host = (host or config.OLLAMA_HOST).rstrip("/")
        self.use_chain = use_chain

    # ------------------------------------------------------------------ #
    # Cloud (OpenAI-compatible) path
    # ------------------------------------------------------------------ #
    def _chat_api(self, messages: list[dict[str, str]]) -> str:
        """Call an OpenAI-compatible chat endpoint. Raises on any failure.

        Rate limits (429) are transient, so they are retried with backoff —
        honouring ``Retry-After`` when the provider sends it — before giving up
        and letting the caller fall back to the local model. Without this, a
        free-tier tokens-per-minute cap would silently push most of a long
        evaluation onto the local model and mix provenance in the results.
        """
        url = f"{config.GENERATION_API_BASE_URL.rstrip('/')}/chat/completions"
        payload = {
            "model": config.GENERATION_API_MODEL,
            "messages": messages,
            "temperature": config.GENERATION_TEMPERATURE,
            "max_tokens": config.GENERATION_MAX_TOKENS,
        }
        headers = {"Authorization": f"Bearer {config.GENERATION_API_KEY}",
                   "Content-Type": "application/json"}

        last_err = ""
        for attempt in range(config.GENERATION_API_MAX_RETRIES + 1):
            try:
                resp = requests.post(url, headers=headers, json=payload,
                                     timeout=config.GENERATION_API_TIMEOUT)
            except requests.RequestException as exc:
                raise GenerationError(f"API request failed: {exc}") from exc

            if resp.status_code == 429 or resp.status_code >= 500:
                last_err = f"API error {resp.status_code}: {resp.text[:200]}"
                if attempt >= config.GENERATION_API_MAX_RETRIES:
                    break
                # Daily quota (resets in minutes-hours): don't stall the query, go local now.
                if _is_quota_exhausted(resp):
                    _log.info("Provider quota exhausted; using local model now")
                    break
                wait = min(
                    _retry_after_seconds(resp) or
                    config.GENERATION_API_BACKOFF_BASE * (2 ** attempt),
                    config.GENERATION_API_MAX_WAIT)
                _log.info("Provider busy (%s); retrying in %.1fs "
                          "(attempt %d/%d)", resp.status_code, wait,
                          attempt + 1, config.GENERATION_API_MAX_RETRIES)
                time.sleep(wait)
                continue

            if resp.status_code >= 400:
                raise GenerationError(
                    f"API error {resp.status_code}: {resp.text[:200]}")
            try:
                text = resp.json()["choices"][0]["message"]["content"]
            except (KeyError, IndexError, ValueError) as exc:
                raise GenerationError(
                    f"Unexpected API response shape: {exc}") from exc
            if not (text or "").strip():
                raise GenerationError("API returned an empty completion")
            return text

        raise GenerationError(last_err or "API retries exhausted")

    def _chat_ollama(self, messages: list[dict[str, str]]) -> str:
        url = f"{self.host}/api/chat"
        try:
            resp = requests.post(
                url,
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": config.GENERATION_TEMPERATURE,
                        "num_predict": config.GENERATION_MAX_TOKENS,
                    },
                },
                timeout=config.HTTP_TIMEOUT_SECONDS * 8,
            )
        except requests.RequestException as exc:
            raise GenerationError(
                f"Could not reach Ollama chat at {self.host}: {exc}"
            ) from exc
        if resp.status_code == 404:
            raise GenerationError(
                f"Chat model {self.model!r} not found. Run: ollama pull {self.model}"
            )
        if resp.status_code >= 400:
            raise GenerationError(f"Ollama chat error {resp.status_code}: "
                                  f"{resp.text[:200]}")
        return resp.json().get("message", {}).get("content", "")

    def _try_alternate_models(self, messages) -> "AnswerResult | None":
        """Retry on the other models in ``GENERATION_ROTATION``.

        Free-tier token budgets are per model, so one model being out of daily
        quota says nothing about the next. Returns ``None`` if every alternate
        is also exhausted, leaving the caller to fall back locally.
        """
        original = config.GENERATION_API_MODEL
        for alt in config.GENERATION_ROTATION:
            if alt == original:
                continue
            try:
                config.GENERATION_API_MODEL = alt
                text = self._chat_api(messages)
            except Exception:  # noqa: BLE001 - this one is out too; try the next
                continue
            finally:
                config.GENERATION_API_MODEL = original
            if text and text.strip():
                _log.info("Rotated generation to %s (previous model out of quota)",
                          alt)
                return AnswerResult(
                    text=text, cited_chunk_ids=extract_citations(text),
                    model=alt, fell_back=True)
        return None

    def answer(
        self,
        question: str,
        chunks: list[dict[str, Any]],
        coverage_since: str = "",
        coverage_until: str = "",
    ) -> AnswerResult:
        messages = build_messages(question, chunks, coverage_since, coverage_until)

        # Cloud first when configured; fall back to local on ANY failure so the
        # app never goes down because a third-party provider did.
        if config.api_generation_enabled():
            try:
                if config.GENERATION_API_THROTTLE:
                    time.sleep(config.GENERATION_API_THROTTLE)
                text = self._chat_api(messages)
                return AnswerResult(
                    text=text, cited_chunk_ids=extract_citations(text),
                    model=config.GENERATION_API_MODEL,
                )
            except Exception as exc:  # noqa: BLE001 - fall back on anything
                if not config.GENERATION_API_FALLBACK:
                    raise
                reason = str(exc)[:200]
                _log.warning("Cloud generation failed: %s", reason)

                # Quota exhaustion is CHEAP to fail over from: the provider
                # rejects in milliseconds, unlike a timeout. Groq bills tokens
                # per DAY *per model*, so when one model's daily budget is gone
                # the others are still full — rotating to the next one keeps the
                # answer on a 70B-class cloud model instead of dropping to the
                # local 7B. This is safe on the interactive path precisely
                # because a 429 costs no wall-clock time (contrast HANDOFF 3.4,
                # which is about *waiting* on a provider, not switching away).
                if _is_quota_or_rate_error(exc):
                    alt = self._try_alternate_models(messages)
                    if alt is not None:
                        alt.fallback_reason = reason
                        return alt

                # Offline evaluation only — see the class docstring. The
                # interactive path skips this entirely and goes straight to
                # local, because each attempt here costs a full timeout.
                if self.use_chain:
                    import providers

                    prompt = _messages_to_prompt(messages)
                    for spec in config.GENERATION_CHAIN[1:]:
                        try:
                            text, used = providers.chat_chain(
                                [spec], prompt,
                                temperature=config.GENERATION_TEMPERATURE)
                        except Exception:  # noqa: BLE001 - try the next link
                            continue
                        # chat_chain appends ollama itself; only accept a real
                        # cloud answer here so the local path stays the single
                        # explicit fallback below.
                        if text.strip() and not used.startswith("ollama"):
                            _log.info("Generation fell through to %s", used)
                            return AnswerResult(
                                text=text,
                                cited_chunk_ids=extract_citations(text),
                                model=used, fell_back=True,
                                fallback_reason=reason)

                _log.warning("Using local model")
                text = self._chat_ollama(messages)
                return AnswerResult(
                    text=text, cited_chunk_ids=extract_citations(text),
                    model=self.model, fell_back=True, fallback_reason=reason,
                )

        text = self._chat_ollama(messages)
        return AnswerResult(
            text=text,
            cited_chunk_ids=extract_citations(text),
            model=self.model,
        )
