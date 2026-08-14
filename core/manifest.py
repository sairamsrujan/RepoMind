"""Read, write, validate, and reuse-check ``manifest.json``.

One manifest lives per repository and drives the central reuse-vs-rebuild
decision. The schema is written by ``write_manifest`` below.

Reuse rule: an index is reusable ONLY if ``schema_version``,
``embedding_model``, ``embedding_dim`` (when known), and ``chunker_version``
all match current config. Otherwise it is stale and must be rebuilt.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config
from core.repo_url import RepoRef

# Status values the pipeline moves through.
VALID_STATUSES = {
    "pending", "fetching", "chunking", "embedding", "indexing", "ready", "failed",
}

_REQUIRED_TOP_LEVEL = {
    "schema_version", "repo", "coverage", "pipeline", "stats",
    "created_at", "updated_at", "status",
}


class ManifestError(ValueError):
    """Raised when a manifest is missing, malformed, or invalid."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_manifest(ref: RepoRef, url: str = "") -> dict[str, Any]:
    """Build a fresh manifest skeleton for a repository (status=pending)."""
    now = _utc_now_iso()
    return {
        "schema_version": config.SCHEMA_VERSION,
        "repo": {
            "owner": ref.owner,
            "name": ref.name,
            "github_id": 0,
            "slug": ref.slug,
            "url": url or ref.html_url,
            "default_branch": "",
            "first_commit_date": "",
        },
        "coverage": {
            "since": "",
            "until": "",
            "commits": {"count": 0, "last_sha": ""},
            "prs": {"count": 0, "last_number": 0},
            "issues": {"count": 0, "last_number": 0},
            "releases": {"count": 0},
        },
        "pipeline": {
            "embedding_model": config.EMBEDDING_MODEL,
            "embedding_dim": 0,
            "chunker_version": config.CHUNKER_VERSION,
            "reranker_model": config.RERANKER_MODEL,
            "nli_model": config.NLI_MODEL,
        },
        "stats": {"chunks": 0, "contributors": 0, "index_bytes": 0},
        "created_at": now,
        "updated_at": now,
        "status": "pending",
    }


def read_manifest(path: str | os.PathLike) -> dict[str, Any]:
    """Load and parse a manifest.json file."""
    p = Path(path)
    if not p.exists():
        raise ManifestError(f"Manifest not found: {p}")
    try:
        with p.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"Manifest is not valid JSON ({p}): {exc}") from exc
    if not isinstance(data, dict):
        raise ManifestError(f"Manifest root must be a JSON object: {p}")
    return data


def write_manifest(path: str | os.PathLike, manifest: dict[str, Any]) -> None:
    """Atomically write a manifest, refreshing ``updated_at``."""
    manifest = dict(manifest)
    manifest["updated_at"] = _utc_now_iso()
    validate_manifest(manifest)  # never persist an invalid manifest
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: temp file in the same dir, then os.replace.
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".manifest.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Validate a manifest against the schema. Raises :class:`ManifestError`."""
    if not isinstance(manifest, dict):
        raise ManifestError("Manifest must be a dict")
    missing = _REQUIRED_TOP_LEVEL - set(manifest)
    if missing:
        raise ManifestError(f"Manifest missing keys: {sorted(missing)}")
    if not isinstance(manifest["schema_version"], int):
        raise ManifestError("schema_version must be an int")
    if manifest["status"] not in VALID_STATUSES:
        raise ManifestError(
            f"Invalid status {manifest['status']!r}; "
            f"expected one of {sorted(VALID_STATUSES)}"
        )
    repo = manifest["repo"]
    for key in ("owner", "name", "slug"):
        if not repo.get(key):
            raise ManifestError(f"repo.{key} is required and non-empty")
    pipeline = manifest["pipeline"]
    for key in ("embedding_model", "chunker_version"):
        if key not in pipeline:
            raise ManifestError(f"pipeline.{key} is required")


def is_reusable(manifest: dict[str, Any]) -> tuple[bool, str]:
    """Decide whether an existing index can be reused as-is.

    Returns ``(reusable, reason)``. ``reason`` explains a rebuild trigger.
    """
    try:
        validate_manifest(manifest)
    except ManifestError as exc:
        return False, f"invalid manifest: {exc}"

    if manifest.get("status") != "ready":
        return False, f"index not ready (status={manifest.get('status')!r})"

    if manifest["schema_version"] != config.SCHEMA_VERSION:
        return False, (
            f"schema_version {manifest['schema_version']} != "
            f"current {config.SCHEMA_VERSION}"
        )

    pipeline = manifest["pipeline"]
    if pipeline.get("embedding_model") != config.EMBEDDING_MODEL:
        return False, (
            f"embedding_model {pipeline.get('embedding_model')!r} != "
            f"current {config.EMBEDDING_MODEL!r}"
        )
    if pipeline.get("chunker_version") != config.CHUNKER_VERSION:
        return False, (
            f"chunker_version {pipeline.get('chunker_version')} != "
            f"current {config.CHUNKER_VERSION}"
        )
    # embedding_dim only meaningful once an index was actually built (>0).
    dim = pipeline.get("embedding_dim", 0)
    if dim and dim <= 0:
        return False, "embedding_dim is non-positive"

    return True, "reusable"
