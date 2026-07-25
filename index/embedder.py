"""Batched embedding client for the local Ollama embedding endpoint.

Uses Ollama's ``/api/embed`` batch endpoint. Embeddings are the one place the
pipeline touches a model server, but it is entirely local (no third-party API).
"""
from __future__ import annotations

from typing import Callable

import requests

import config

ProgressFn = Callable[[str], None]


class EmbeddingError(RuntimeError):
    pass


class OllamaEmbedder:
    """Embeds text via a local Ollama model, batched for throughput."""

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        batch_size: int | None = None,
    ):
        self.model = model or config.EMBEDDING_MODEL
        self.host = (host or config.OLLAMA_HOST).rstrip("/")
        self.batch_size = batch_size or config.EMBED_BATCH_SIZE
        self._dim: int | None = None

    # ------------------------------------------------------------------ #
    def _post_embed(self, inputs: list[str]) -> list[list[float]]:
        url = f"{self.host}/api/embed"
        try:
            resp = requests.post(
                url,
                json={"model": self.model, "input": inputs},
                timeout=config.HTTP_TIMEOUT_SECONDS * 4,
            )
        except requests.RequestException as exc:
            raise EmbeddingError(
                f"Could not reach Ollama at {self.host}: {exc}. "
                f"Is `ollama serve` running?"
            ) from exc
        if resp.status_code == 404:
            raise EmbeddingError(
                f"Embedding model {self.model!r} not found on Ollama. "
                f"Run: ollama pull {self.model}"
            )
        if resp.status_code >= 400:
            raise EmbeddingError(f"Ollama embed error {resp.status_code}: "
                                 f"{resp.text[:200]}")
        data = resp.json()
        embeddings = data.get("embeddings")
        if not embeddings:
            raise EmbeddingError(f"Ollama returned no embeddings: {data}")
        return embeddings

    # ------------------------------------------------------------------ #
    def embed_batch(
        self, texts: list[str], progress: ProgressFn | None = None
    ) -> list[list[float]]:
        """Embed many texts, chunked into batches. Preserves order."""
        out: list[list[float]] = []
        total = len(texts)
        for start in range(0, total, self.batch_size):
            batch = texts[start:start + self.batch_size]
            vectors = self._post_embed(batch)
            out.extend(vectors)
            if self._dim is None and vectors:
                self._dim = len(vectors[0])
            if progress:
                progress(f"embedded {min(start + self.batch_size, total)}/{total}")
        return out

    def embed_query(self, text: str) -> list[float]:
        vec = self._post_embed([text])[0]
        if self._dim is None:
            self._dim = len(vec)
        return vec

    @property
    def dim(self) -> int:
        """Embedding dimensionality (probes the model once if unknown)."""
        if self._dim is None:
            self.embed_query("dimension probe")
        return int(self._dim)
