"""Cross-encoder reranker over the MMR-selected candidates.

Wraps a sentence-transformers ``CrossEncoder``. Default model is
``BAAI/bge-reranker-v2-m3``; if it cannot be loaded (offline, disk, etc.) it
falls back to the lighter ``cross-encoder/ms-marco-MiniLM-L-6-v2``. Model
loading is lazy and cached per process.
"""
from __future__ import annotations

from typing import Any

import config

_MODEL_CACHE: dict[str, Any] = {}


def _load_cross_encoder(model_name: str):
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name, max_length=512)
    _MODEL_CACHE[model_name] = model
    return model


class Reranker:
    """Reranks (query, chunk) pairs by cross-encoder relevance score."""

    def __init__(self, model_name: str | None = None, allow_fallback: bool = True):
        self.model_name = model_name or config.RERANKER_MODEL
        self.allow_fallback = allow_fallback
        self._model = None

    def _model_lazy(self):
        if self._model is not None:
            return self._model
        try:
            self._model = _load_cross_encoder(self.model_name)
        except Exception:
            if not self.allow_fallback:
                raise
            self.model_name = config.RERANKER_MODEL_FALLBACK
            self._model = _load_cross_encoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return ``chunks`` reordered by relevance, truncated to ``top_k``.

        Each returned chunk gains a ``rerank_score`` field.
        """
        k = top_k or config.FINAL_TOP_K
        if not chunks:
            return []
        model = self._model_lazy()
        pairs = [[query, c["text"]] for c in chunks]
        scores = model.predict(pairs, show_progress_bar=False)
        for c, s in zip(chunks, scores):
            c["rerank_score"] = float(s)
        ranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
        return ranked[:k]
