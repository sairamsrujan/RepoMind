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

_CITATION_RE = re.compile(r"\[([A-Za-z0-9_#\-]+)\]")
_log = logging.getLogger(__name__)


def _retry_after_seconds(resp) -> float | None:
    """Parse a Retry-After header (seconds form) if the provider sent one."""
    raw = (resp.headers or {}).get("Retry-After")
    if not raw:
        return None
    try:
        return max(0.0, min(float(raw), 60.0))
    except (TypeError, ValueError):
        return None


@dataclass
class AnswerResult:
    text: str
    cited_chunk_ids: list[str] = field(default_factory=list)
    model: str = ""
    # Set when cloud generation was attempted but the local model answered.
    fell_back: bool = False
    fallback_reason: str = ""


def extract_citations(text: str) -> list[str]:
    """Return the ordered, deduped chunk_ids cited inline in ``text``."""
    seen: set[str] = set()
    out: list[str] = []
    for m in _CITATION_RE.finditer(text or ""):
        cid = m.group(1)
        if cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


class GenerationError(RuntimeError):
    pass


class Answerer:
    """Grounded answering via local Ollama, or a hosted model with fallback."""

    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or config.GENERATION_MODEL
        self.host = (host or config.OLLAMA_HOST).rstrip("/")

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
                wait = _retry_after_seconds(resp) or min(
                    config.GENERATION_API_BACKOFF_BASE * (2 ** attempt), 60.0)
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
                _log.warning("Cloud generation failed, using local model: %s",
                             reason)
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
