"""Build and load the per-repository indexes: Chroma (dense) + BM25 (sparse).

One Chroma collection per repository stores every chunk's embedding, document
text, and metadata (for pre-filtering). A parallel BM25 index over the same
chunk set is pickled alongside. After a build, the manifest is updated with
pipeline info, coverage, and stats.
"""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any, Callable

import chromadb
from chromadb.config import Settings

import config
from core import manifest as manifest_mod
from core.context import RepositoryContext
from index.embedder import OllamaEmbedder

ProgressFn = Callable[[str], None]

_TOKEN_RE = re.compile(r"[A-Za-z0-9_#]+")


def bm25_tokenize(text: str) -> list[str]:
    """Lowercase word/identifier tokenizer shared by build and query sides."""
    return _TOKEN_RE.findall((text or "").lower())


# --------------------------------------------------------------------------- #
# Chroma helpers
# --------------------------------------------------------------------------- #
def _chroma_client(ctx: RepositoryContext) -> chromadb.ClientAPI:
    ctx.chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(ctx.chroma_dir),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )


def _to_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    """Chroma metadata must be scalar-valued; serialize lists as JSON strings."""
    return {
        "source_type": chunk["source_type"],
        "ref_id": str(chunk["ref_id"]),
        "title": chunk.get("title", "")[:500],
        "author": chunk.get("author", ""),
        "date": chunk.get("date", ""),
        "github_url": chunk.get("github_url", ""),
        "linked_refs": json.dumps(chunk.get("linked_refs", [])),
        "token_count": int(chunk.get("token_count", 0)),
    }


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build_index(
    ctx: RepositoryContext,
    chunks: list[dict[str, Any]],
    embedder: OllamaEmbedder | None = None,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Embed chunks, build Chroma + BM25 indexes, update the manifest.

    Returns a small stats dict. Rebuilds from scratch (drops any existing
    collection) so a config change produces a clean index.
    """
    if not chunks:
        raise ValueError("No chunks to index")
    embedder = embedder or OllamaEmbedder()

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    if progress:
        progress(f"embedding {len(texts)} chunks with {embedder.model}")
    vectors = embedder.embed_batch(texts, progress=progress)
    dim = len(vectors[0]) if vectors else 0

    # --- Chroma (dense) --- #
    client = _chroma_client(ctx)
    try:
        client.delete_collection(ctx.collection_name)
    except Exception:
        pass
    collection = client.create_collection(
        name=ctx.collection_name, metadata={"hnsw:space": "cosine"}
    )
    metadatas = [_to_metadata(c) for c in chunks]
    # Add in batches (Chroma has a per-call cap).
    B = 512
    for start in range(0, len(ids), B):
        collection.add(
            ids=ids[start:start + B],
            embeddings=vectors[start:start + B],
            documents=texts[start:start + B],
            metadatas=metadatas[start:start + B],
        )
    if progress:
        progress("dense index built")

    # --- BM25 (sparse) --- #
    tokenized = [bm25_tokenize(t) for t in texts]
    bm25_payload = {
        "chunk_ids": ids,
        "tokenized": tokenized,
        "chunks": {c["chunk_id"]: c for c in chunks},
    }
    ctx.bm25_path.parent.mkdir(parents=True, exist_ok=True)
    with ctx.bm25_path.open("wb") as fh:
        pickle.dump(bm25_payload, fh)
    if progress:
        progress("sparse index built")

    stats = {
        "chunks": len(chunks),
        "contributors": len({c.get("author", "") for c in chunks if c.get("author")}),
        "embedding_dim": dim,
        "index_bytes": _dir_size_bytes(ctx.index_dir),
    }
    _update_manifest_after_build(ctx, chunks, stats)
    return stats


def _update_manifest_after_build(
    ctx: RepositoryContext, chunks: list[dict], stats: dict
) -> None:
    """Refresh (or create) the manifest with pipeline/coverage/stats + ready."""
    if ctx.manifest_path.exists():
        m = manifest_mod.read_manifest(ctx.manifest_path)
    else:
        from core.repo_url import RepoRef
        m = manifest_mod.new_manifest(RepoRef(ctx.ref.owner, ctx.ref.name))

    m["pipeline"]["embedding_model"] = config.EMBEDDING_MODEL
    m["pipeline"]["embedding_dim"] = stats["embedding_dim"]
    m["pipeline"]["chunker_version"] = config.CHUNKER_VERSION
    m["pipeline"]["reranker_model"] = config.RERANKER_MODEL
    m["pipeline"]["nli_model"] = config.NLI_MODEL

    dates = sorted(c["date"] for c in chunks if c.get("date"))
    if dates:
        m["coverage"]["since"] = m["coverage"].get("since") or dates[0]
        m["coverage"]["until"] = dates[-1]

    def _count(stype):
        return sum(1 for c in chunks if c["source_type"] == stype)

    m["coverage"]["commits"]["count"] = _count("commit")
    m["coverage"]["prs"]["count"] = _count("pr")
    m["coverage"]["issues"]["count"] = _count("issue")
    m["coverage"]["releases"]["count"] = _count("release")

    m["stats"]["chunks"] = stats["chunks"]
    m["stats"]["contributors"] = stats["contributors"]
    m["stats"]["index_bytes"] = stats["index_bytes"]
    m["status"] = "ready"
    manifest_mod.write_manifest(ctx.manifest_path, m)


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def load_collection(ctx: RepositoryContext):
    """Return the persisted Chroma collection for ``ctx``."""
    client = _chroma_client(ctx)
    return client.get_collection(ctx.collection_name)


def load_bm25_payload(ctx: RepositoryContext) -> dict[str, Any]:
    """Return the pickled BM25 payload (ids, tokenized corpus, chunk map)."""
    with ctx.bm25_path.open("rb") as fh:
        return pickle.load(fh)


def index_exists(ctx: RepositoryContext) -> bool:
    return ctx.bm25_path.exists() and ctx.chroma_dir.exists()
