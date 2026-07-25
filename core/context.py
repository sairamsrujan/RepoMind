"""RepositoryContext: the one abstraction that keeps every module repo-agnostic.

Given a :class:`RepoRef`, it owns the ``repositories/<slug>/`` directory tree
and exposes typed paths so no other module ever builds a path by hand.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import config
from core.repo_url import RepoRef


@dataclass(frozen=True)
class RepositoryContext:
    """Filesystem layout + identity for a single indexed repository."""

    ref: RepoRef
    base_dir: Path

    # ---- construction ---------------------------------------------------- #
    @classmethod
    def for_ref(
        cls, ref: RepoRef, repositories_dir: Path | None = None
    ) -> "RepositoryContext":
        root = repositories_dir or config.REPOSITORIES_DIR
        return cls(ref=ref, base_dir=Path(root) / ref.slug)

    @property
    def slug(self) -> str:
        return self.ref.slug

    # ---- top-level files ------------------------------------------------- #
    @property
    def manifest_path(self) -> Path:
        return self.base_dir / "manifest.json"

    @property
    def status_path(self) -> Path:
        return self.base_dir / "job_status.json"

    @property
    def lock_path(self) -> Path:
        return self.base_dir / ".lock"

    # ---- raw data -------------------------------------------------------- #
    @property
    def raw_dir(self) -> Path:
        return self.base_dir / "raw"

    @property
    def raw_commits_dir(self) -> Path:
        return self.raw_dir / "commits"

    @property
    def raw_prs_dir(self) -> Path:
        return self.raw_dir / "prs"

    @property
    def raw_issues_dir(self) -> Path:
        return self.raw_dir / "issues"

    @property
    def raw_releases_dir(self) -> Path:
        return self.raw_dir / "releases"

    @property
    def checkpoint_path(self) -> Path:
        return self.raw_dir / "checkpoint.json"

    # ---- processed data -------------------------------------------------- #
    @property
    def chunks_dir(self) -> Path:
        return self.base_dir / "chunks"

    @property
    def chunks_path(self) -> Path:
        return self.chunks_dir / "chunks.jsonl"

    @property
    def links_path(self) -> Path:
        return self.chunks_dir / "links.json"

    @property
    def embeddings_dir(self) -> Path:
        return self.base_dir / "embeddings"

    # ---- index ----------------------------------------------------------- #
    @property
    def index_dir(self) -> Path:
        return self.base_dir / "index"

    @property
    def chroma_dir(self) -> Path:
        return self.index_dir / "chroma"

    @property
    def bm25_path(self) -> Path:
        return self.index_dir / "bm25.pkl"

    @property
    def collection_name(self) -> str:
        """Chroma collection name (one per repository)."""
        # Chroma names: 3-63 chars, alnum/._-, start & end alnum.
        name = f"repo_{self.slug}".replace(".", "_")
        return name[:63]

    # ---- lifecycle ------------------------------------------------------- #
    def ensure_dirs(self) -> "RepositoryContext":
        """Create the full directory tree for this repository."""
        for d in (
            self.base_dir,
            self.raw_commits_dir,
            self.raw_prs_dir,
            self.raw_issues_dir,
            self.raw_releases_dir,
            self.chunks_dir,
            self.embeddings_dir,
            self.index_dir,
            self.chroma_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def exists(self) -> bool:
        """True if this repository has at least a manifest on disk."""
        return self.manifest_path.exists()
