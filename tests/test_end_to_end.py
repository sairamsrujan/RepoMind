"""End-to-end pipeline over fixture data (Phase 7 walkthrough, offline).

chunk -> link -> index -> retrieve -> answer -> guard. Uses local Ollama for
embeddings and generation; skips if Ollama is unreachable.
"""
from __future__ import annotations

import pytest
import requests

import config
from guard.reference_validator import validate_references
from index import vector_store
from process import chunker, linker


def _ollama_up() -> bool:
    try:
        requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=3)
        return True
    except requests.RequestException:
        return False


pytestmark = pytest.mark.skipif(not _ollama_up(), reason="Ollama not running")


def test_full_pipeline_answers_and_guards(fixture_repo):
    ctx = fixture_repo
    chunker.chunk_repository(ctx)
    linker.link_repository(ctx)
    chunks = chunker.load_chunks(ctx)
    vector_store.build_index(ctx, chunks)

    from retrieval.retriever import Retriever

    retriever = Retriever(ctx)
    retrieved = retriever.retrieve("How was the startup crash fixed?", top_k=6)
    assert retrieved
    ids = {c["chunk_id"] for c in retrieved}
    # The fix lives in PR #101 / commit c0ffee1 — at least one must surface.
    assert any(i.startswith(("pr_101", "commit_c0ffee1")) for i in ids)

    from generation.answerer import Answerer

    result = Answerer().answer(
        "How was the startup crash fixed?", retrieved,
        "2024-03-01", "2024-03-10",
    )
    assert result.text.strip()

    # Every inline citation must resolve to a retrieved chunk.
    report = validate_references(result.text, retrieved)
    assert report.is_valid, f"fabricated citations: {report.invalid_citations}"
