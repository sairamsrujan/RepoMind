"""Phase 4: index build, reuse-vs-rebuild, and dense/sparse index integrity.

Uses the real local Ollama embedding endpoint; skips if it is unreachable.
"""
from __future__ import annotations

import pytest
import requests

import config
from core import manifest as manifest_mod
from index import vector_store
from index.embedder import OllamaEmbedder
from process import chunker


def _ollama_up() -> bool:
    try:
        requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=3)
        return True
    except requests.RequestException:
        return False


pytestmark = pytest.mark.skipif(
    not _ollama_up(), reason="Ollama not running"
)


def test_build_index_and_reuse(fixture_repo, monkeypatch):
    ctx = fixture_repo
    n = chunker.chunk_repository(ctx)
    chunks = chunker.load_chunks(ctx)
    assert len(chunks) == n

    stats = vector_store.build_index(ctx, chunks, OllamaEmbedder())
    assert stats["chunks"] == n
    assert stats["embedding_dim"] > 0
    assert stats["contributors"] >= 1
    assert stats["index_bytes"] > 0

    # Dense index populated.
    coll = vector_store.load_collection(ctx)
    assert coll.count() == n

    # Sparse index populated and aligned.
    payload = vector_store.load_bm25_payload(ctx)
    assert len(payload["chunk_ids"]) == n
    assert len(payload["tokenized"]) == n

    # Manifest ready + reusable under current config.
    m = manifest_mod.read_manifest(ctx.manifest_path)
    assert m["status"] == "ready"
    assert m["pipeline"]["embedding_dim"] == stats["embedding_dim"]
    ok, reason = manifest_mod.is_reusable(m)
    assert ok, reason

    # Changing the embedding model must trigger a rebuild (not reuse).
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "some-other-embed:1.0")
    ok2, reason2 = manifest_mod.is_reusable(m)
    assert not ok2 and "embedding_model" in reason2


def test_reindex_replaces_collection(fixture_repo):
    ctx = fixture_repo
    chunker.chunk_repository(ctx)
    chunks = chunker.load_chunks(ctx)
    vector_store.build_index(ctx, chunks, OllamaEmbedder())
    first = vector_store.load_collection(ctx).count()
    # Rebuild again; count should stay the same (no duplication).
    vector_store.build_index(ctx, chunks, OllamaEmbedder())
    second = vector_store.load_collection(ctx).count()
    assert first == second == len(chunks)
