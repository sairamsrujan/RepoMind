"""Phase 5: MMR diversity, reranker reordering, and BM25 identifier recall."""
from __future__ import annotations

import pytest
import requests

import config
from retrieval.mmr import mmr_select


def _ollama_up() -> bool:
    try:
        requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=3)
        return True
    except requests.RequestException:
        return False


# --------------------------------------------------------------------------- #
# MMR — pure, always runs
# --------------------------------------------------------------------------- #
def test_mmr_reduces_near_duplicates():
    q = [1.0, 0.0, 0.0]
    # A and A2 are near-identical (redundant); B is relevant but diverse.
    A = [0.9, 0.436, 0.0]
    A2 = [0.9, 0.436, 0.0]
    B = [0.8, 0.0, 0.6]
    ids = ["A", "A2", "B"]
    vecs = [A, A2, B]

    mmr_top2 = mmr_select(q, ids, vecs, lambda_mult=0.5, top_n=2)
    # Pure relevance would return [A, A2] (the near-duplicate). MMR should
    # instead surface the diverse B and drop the redundant A2.
    assert mmr_top2 == ["A", "B"]
    assert "A2" not in mmr_top2


def test_mmr_lambda_one_is_pure_relevance():
    q = [1.0, 0.0, 0.0]
    ids = ["A", "A2", "B"]
    vecs = [[0.9, 0.436, 0.0], [0.9, 0.436, 0.0], [0.8, 0.0, 0.6]]
    # lambda=1 ignores diversity -> ranks purely by relevance (A, A2 first).
    top2 = mmr_select(q, ids, vecs, lambda_mult=1.0, top_n=2)
    assert set(top2) == {"A", "A2"}


# --------------------------------------------------------------------------- #
# Reranker — needs a cross-encoder model
# --------------------------------------------------------------------------- #
def test_reranker_reorders_shuffled_pair():
    from retrieval.reranker import Reranker

    try:
        # Use the light fallback model for a fast test.
        rr = Reranker(model_name=config.RERANKER_MODEL_FALLBACK, allow_fallback=False)
        rr._model_lazy()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"reranker model unavailable: {exc}")

    query = "How was the startup crash fixed?"
    # Deliberately shuffled: irrelevant first, relevant second.
    chunks = [
        {"chunk_id": "irrelevant",
         "text": "The weather in Paris is pleasant during spring."},
        {"chunk_id": "relevant",
         "text": "We fixed the null pointer crash on startup by null-checking "
                 "the configuration before use."},
    ]
    ranked = rr.rerank(query, chunks, top_k=2)
    assert ranked[0]["chunk_id"] == "relevant"
    assert ranked[0]["rerank_score"] > ranked[1]["rerank_score"]


# --------------------------------------------------------------------------- #
# BM25 identifier recall + hybrid retrieve — needs Ollama
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _ollama_up(), reason="Ollama not running")
def test_bm25_recovers_identifier_query(tmp_path):
    import json

    from core.context import RepositoryContext
    from core.repo_url import parse_repo_url
    from index import vector_store
    from retrieval.retriever import Retriever

    ref = parse_repo_url("acme/synthetic")
    ctx = RepositoryContext.for_ref(ref, tmp_path / "repositories").ensure_dirs()

    # A corpus of generic chunks + one carrying a rare identifier token.
    texts = [
        "General discussion about improving documentation and examples.",
        "Refactoring the widget module for readability.",
        "Adding unit tests for the caching layer.",
        "Investigating memory usage under heavy load.",
        "Tracking work for ticket ZQX-9987 about the export pipeline bug.",
    ]
    chunks = []
    for i, t in enumerate(texts):
        chunks.append({
            "chunk_id": f"issue_{i}", "source_type": "issue", "ref_id": str(i),
            "title": t[:40], "author": "dev", "date": "2024-05-0%d" % (i + 1),
            "github_url": f"https://github.com/acme/synthetic/issues/{i}",
            "linked_refs": [], "text": t, "token_count": len(t.split()),
        })
    ctx.chunks_dir.mkdir(parents=True, exist_ok=True)
    with ctx.chunks_path.open("w") as fh:
        for c in chunks:
            fh.write(json.dumps(c) + "\n")

    vector_store.build_index(ctx, chunks)
    r = Retriever(ctx)

    # The identifier is a rare token dense search may miss but BM25 nails.
    sparse_ids = r.sparse_search("ZQX-9987 export bug", k=5, filters=None)
    assert "issue_4" in sparse_ids

    # End-to-end retrieve (with MMR + reranker) should surface it too.
    results = r.retrieve("ZQX-9987 export bug", top_k=3)
    assert any(c["chunk_id"] == "issue_4" for c in results)


@pytest.mark.skipif(not _ollama_up(), reason="Ollama not running")
def test_retrieve_trace_populates_and_preserves_result(tmp_path):
    """Phase A: passing a `trace` dict records stage counts/timings but must not
    change the returned chunks (byte-for-byte behaviour preserved)."""
    import json

    from core.context import RepositoryContext
    from core.repo_url import parse_repo_url
    from index import vector_store
    from retrieval.retriever import Retriever

    ref = parse_repo_url("acme/tracetest")
    ctx = RepositoryContext.for_ref(ref, tmp_path / "repositories").ensure_dirs()
    chunks = []
    for i, t in enumerate([
        "Added retry logic to the network client for flaky connections.",
        "Refactored the settings page for accessibility.",
        "Fixed a race condition in the background worker.",
        "Documented the plugin API with examples.",
    ]):
        chunks.append({
            "chunk_id": f"pr_{i}", "source_type": "pr", "ref_id": str(i),
            "title": t[:40], "author": "dev", "date": "2024-05-0%d" % (i + 1),
            "github_url": f"https://github.com/acme/tracetest/pull/{i}",
            "linked_refs": [], "text": t, "token_count": len(t.split()),
        })
    ctx.chunks_dir.mkdir(parents=True, exist_ok=True)
    with ctx.chunks_path.open("w") as fh:
        for c in chunks:
            fh.write(json.dumps(c) + "\n")
    vector_store.build_index(ctx, chunks)
    r = Retriever(ctx)

    q = "how were flaky network connections handled?"
    without = [c["chunk_id"] for c in r.retrieve(q, top_k=3)]
    trace: dict = {}
    with_trace = [c["chunk_id"] for c in r.retrieve(q, top_k=3, trace=trace)]

    # Same result either way.
    assert without == with_trace
    # Trace fully populated.
    for key in ("dense_count", "sparse_count", "rrf_count", "mmr_count",
                "final_count", "retrieval_ms", "rerank_ms"):
        assert key in trace, f"trace missing {key}"
    assert trace["final_count"] == len(with_trace)
    assert trace["retrieval_ms"] >= 0.0 and trace["rerank_ms"] >= 0.0
