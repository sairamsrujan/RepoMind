"""Maximal Marginal Relevance (MMR) re-ranking for diversity.

Given a query vector and candidate vectors, iteratively pick the candidate that
best balances relevance to the query against redundancy with already-picked
candidates. ``lambda`` (default 0.5) trades off the two.
"""
from __future__ import annotations

import numpy as np

import config


def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T


def mmr_select(
    query_vec: list[float],
    candidate_ids: list[str],
    candidate_vecs: list[list[float]],
    lambda_mult: float | None = None,
    top_n: int | None = None,
) -> list[str]:
    """Return candidate ids ordered by MMR (most relevant+diverse first)."""
    if not candidate_ids:
        return []
    lam = config.MMR_LAMBDA if lambda_mult is None else lambda_mult
    k = top_n or config.MMR_TOP_N
    k = min(k, len(candidate_ids))

    docs = np.asarray(candidate_vecs, dtype=np.float32)
    q = np.asarray([query_vec], dtype=np.float32)

    rel = _cosine_matrix(q, docs)[0]        # (n,) query-doc similarity
    sim = _cosine_matrix(docs, docs)        # (n,n) doc-doc similarity

    selected: list[int] = []
    remaining = set(range(len(candidate_ids)))

    # Seed with the most relevant candidate.
    first = int(np.argmax(rel))
    selected.append(first)
    remaining.discard(first)

    while remaining and len(selected) < k:
        best_idx, best_score = None, -np.inf
        for i in remaining:
            redundancy = max(sim[i][j] for j in selected)
            score = lam * rel[i] - (1.0 - lam) * redundancy
            if score > best_score:
                best_score, best_idx = score, i
        selected.append(best_idx)
        remaining.discard(best_idx)

    return [candidate_ids[i] for i in selected]
