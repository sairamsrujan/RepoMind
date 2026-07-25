"""Hybrid retrieval: dense (Chroma) + sparse (BM25), fused with RRF.

Pipeline for a query:
  1. Apply filters to shrink the candidate pool (pre-filter, not post-filter).
  2. Dense search over Chroma embeddings; sparse search over BM25.
  3. Merge both ranked lists with Reciprocal Rank Fusion (RRF), dedup by id.
  4. Optionally diversify with MMR, then optionally rerank with a cross-encoder.

The MMR / reranker stages are toggleable so the evaluation ablation can compare
retrieval-only vs +MMR vs +reranker.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from rank_bm25 import BM25Okapi

import config
from core.context import RepositoryContext
from index.embedder import OllamaEmbedder
from index.vector_store import bm25_tokenize, load_bm25_payload, load_collection
from retrieval.filters import RetrievalFilters, build_chroma_where, make_predicate
from retrieval.mmr import mmr_select
from retrieval.reranker import Reranker


@dataclass
class Retriever:
    """Stateful retriever bound to one repository's indexes."""

    ctx: RepositoryContext
    embedder: OllamaEmbedder = field(default_factory=OllamaEmbedder)
    reranker: Reranker | None = None

    _collection: Any = field(default=None, init=False, repr=False)
    _bm25: Any = field(default=None, init=False, repr=False)
    _ids: list[str] = field(default_factory=list, init=False, repr=False)
    _chunks: dict[str, dict] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._collection = load_collection(self.ctx)
        payload = load_bm25_payload(self.ctx)
        self._ids = payload["chunk_ids"]
        self._chunks = payload["chunks"]
        self._bm25 = BM25Okapi(payload["tokenized"])
        if self.reranker is None:
            self.reranker = Reranker()

    # ------------------------------------------------------------------ #
    # Individual retrievers
    # ------------------------------------------------------------------ #
    def dense_search(
        self, query_vec: list[float], k: int, where: dict | None
    ) -> list[str]:
        res = self._collection.query(
            query_embeddings=[query_vec],
            n_results=min(k, max(1, self._collection.count())),
            where=where,
        )
        ids = res.get("ids", [[]])
        return ids[0] if ids else []

    def sparse_search(
        self, query: str, k: int, filters: RetrievalFilters | None
    ) -> list[str]:
        scores = self._bm25.get_scores(bm25_tokenize(query))
        predicate = make_predicate(filters)
        scored = [
            (self._ids[i], scores[i])
            for i in range(len(self._ids))
            if predicate(self._chunks[self._ids[i]])
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [cid for cid, s in scored[:k] if s > 0]

    # ------------------------------------------------------------------ #
    # Reciprocal Rank Fusion
    # ------------------------------------------------------------------ #
    @staticmethod
    def rrf_fuse(ranked_lists: list[list[str]], pool_size: int) -> list[str]:
        scores: dict[str, float] = {}
        for ranked in ranked_lists:
            for rank, cid in enumerate(ranked, start=1):
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (config.RRF_K + rank)
        fused = sorted(scores, key=lambda c: scores[c], reverse=True)
        return fused[:pool_size]

    # ------------------------------------------------------------------ #
    # Full pipeline
    # ------------------------------------------------------------------ #
    def retrieve(
        self,
        query: str,
        filters: RetrievalFilters | None = None,
        use_mmr: bool = True,
        use_reranker: bool = True,
        top_k: int | None = None,
        trace: dict[str, Any] | None = None,
        dense_k: int | None = None,
        sparse_k: int | None = None,
        use_dense: bool = True,
        use_sparse: bool = True,
    ) -> list[dict[str, Any]]:
        """Return the final top-k chunk dicts for ``query``.

        If ``trace`` is a dict, per-stage candidate counts and timings are
        written into it (opt-in instrumentation for Phase A metrics). When
        ``trace`` is None the returned result is identical to before.

        ``dense_k`` / ``sparse_k`` override the dense / BM25 candidate-pool sizes
        (default ``None`` uses ``config.DENSE_TOP_K`` / ``config.SPARSE_TOP_K``);
        the Phase B adaptive retry uses these to widen the pools.
        """
        k = top_k or config.FINAL_TOP_K
        where = build_chroma_where(filters)
        _t0 = time.perf_counter()
        query_vec = self.embedder.embed_query(query)

        dense = (self.dense_search(query_vec, dense_k or config.DENSE_TOP_K, where)
                 if use_dense else [])
        sparse = (self.sparse_search(query, sparse_k or config.SPARSE_TOP_K, filters)
                  if use_sparse else [])
        pool_ids = self.rrf_fuse([dense, sparse], config.RRF_POOL_SIZE)
        rrf_count = len(pool_ids)
        if not pool_ids:
            if trace is not None:
                trace.update(dense_count=len(dense), sparse_count=len(sparse),
                             rrf_count=0, mmr_count=0, final_count=0,
                             retrieval_ms=(time.perf_counter() - _t0) * 1000.0,
                             rerank_ms=0.0)
            return []

        if use_mmr:
            pool_ids = self._mmr(query_vec, pool_ids)
        mmr_count = len(pool_ids)

        candidates = [dict(self._chunks[cid]) for cid in pool_ids if cid in self._chunks]
        _retrieval_ms = (time.perf_counter() - _t0) * 1000.0

        _t1 = time.perf_counter()
        if use_reranker:
            result = self.reranker.rerank(query, candidates, top_k=k)
        else:
            result = candidates[:k]
        _rerank_ms = (time.perf_counter() - _t1) * 1000.0

        if trace is not None:
            trace.update(dense_count=len(dense), sparse_count=len(sparse),
                         rrf_count=rrf_count, mmr_count=mmr_count,
                         final_count=len(result), retrieval_ms=_retrieval_ms,
                         rerank_ms=_rerank_ms)
        return result

    def _mmr(self, query_vec: list[float], pool_ids: list[str]) -> list[str]:
        got = self._collection.get(ids=pool_ids, include=["embeddings"])
        got_ids = got.get("ids", [])
        got_vecs = got.get("embeddings", [])
        vec_by_id = {i: v for i, v in zip(got_ids, got_vecs)}
        ids, vecs = [], []
        for cid in pool_ids:
            v = vec_by_id.get(cid)
            if v is not None:
                ids.append(cid)
                vecs.append(list(v))
        if not ids:
            return pool_ids
        return mmr_select(query_vec, ids, vecs, top_n=config.MMR_TOP_N)
