"""Generate a grounded answer with the local Ollama chat model."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import requests

import config
from generation.prompt import build_messages

_CITATION_RE = re.compile(r"\[([A-Za-z0-9_#\-]+)\]")


@dataclass
class AnswerResult:
    text: str
    cited_chunk_ids: list[str] = field(default_factory=list)
    model: str = ""


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
    """Wraps the local Ollama chat endpoint for grounded answering."""

    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or config.GENERATION_MODEL
        self.host = (host or config.OLLAMA_HOST).rstrip("/")

    def _chat(self, messages: list[dict[str, str]]) -> str:
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
        text = self._chat(messages)
        return AnswerResult(
            text=text,
            cited_chunk_ids=extract_citations(text),
            model=self.model,
        )
