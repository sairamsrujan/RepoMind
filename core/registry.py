"""Registry of indexed repositories.

Scans ``repositories/`` for any directory containing a manifest, so the UI can
list previously indexed repos, switch between them, and detect whether a given
:class:`RepoRef` is already indexed (and reusable).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import config
from core import manifest as manifest_mod
from core.context import RepositoryContext
from core.repo_url import RepoRef


@dataclass(frozen=True)
class RegistryEntry:
    """A summary row for one indexed repository, read from its manifest."""

    slug: str
    owner: str
    name: str
    status: str
    reusable: bool
    reuse_reason: str
    manifest: dict[str, Any]

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.name}"

    @property
    def chunks(self) -> int:
        return int(self.manifest.get("stats", {}).get("chunks", 0))


def _entry_from_manifest(m: dict[str, Any]) -> RegistryEntry:
    reusable, reason = manifest_mod.is_reusable(m)
    repo = m.get("repo", {})
    return RegistryEntry(
        slug=repo.get("slug", ""),
        owner=repo.get("owner", ""),
        name=repo.get("name", ""),
        status=m.get("status", "unknown"),
        reusable=reusable,
        reuse_reason=reason,
        manifest=m,
    )


def list_repositories(repositories_dir: Path | None = None) -> list[RegistryEntry]:
    """Return all indexed repositories, newest-updated first."""
    root = Path(repositories_dir or config.REPOSITORIES_DIR)
    if not root.exists():
        return []
    entries: list[RegistryEntry] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        mpath = child / "manifest.json"
        if not mpath.exists():
            continue
        try:
            m = manifest_mod.read_manifest(mpath)
        except manifest_mod.ManifestError:
            continue  # skip corrupt manifests rather than crash the registry
        entries.append(_entry_from_manifest(m))
    entries.sort(key=lambda e: e.manifest.get("updated_at", ""), reverse=True)
    return entries


def find(ref: RepoRef, repositories_dir: Path | None = None) -> RegistryEntry | None:
    """Return the registry entry for ``ref`` if it is already indexed."""
    ctx = RepositoryContext.for_ref(ref, repositories_dir)
    if not ctx.manifest_path.exists():
        return None
    try:
        m = manifest_mod.read_manifest(ctx.manifest_path)
    except manifest_mod.ManifestError:
        return None
    return _entry_from_manifest(m)


def is_indexed_and_reusable(
    ref: RepoRef, repositories_dir: Path | None = None
) -> bool:
    """True only if ``ref`` is indexed AND its index is reusable per config."""
    entry = find(ref, repositories_dir)
    return bool(entry and entry.reusable)
