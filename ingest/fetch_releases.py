"""Fetch releases via the GitHub REST API.

Releases are not date-filterable server-side, so all releases are fetched and
the ``until`` bound is applied client-side on the published date. The set is
usually small, so this is cheap.
"""
from __future__ import annotations

import json
from typing import Callable

import config
from core.context import RepositoryContext
from core.repo_url import RepoRef
from ingest.checkpoint import Checkpoint
from ingest.github_client import GitHubClient

ProgressFn = Callable[[str], None]


def _write_page(directory, page: int, data: list) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"page_{page:04d}.json").write_text(
        json.dumps(data), encoding="utf-8"
    )


def fetch_releases(
    client: GitHubClient,
    ctx: RepositoryContext,
    ref: RepoRef,
    since: str,
    until: str,
    checkpoint: Checkpoint,
    progress: ProgressFn | None = None,
) -> int:
    """Fetch releases published within [since, until]; return count."""
    if checkpoint.is_done("releases"):
        return checkpoint.get("releases")["count"]

    url = f"{config.GITHUB_API_URL}/repos/{ref.owner}/{ref.name}/releases"
    state = checkpoint.get("releases")
    start_page = state["page"] + 1
    total = state["count"]

    page = start_page
    while True:
        params = {"per_page": config.GITHUB_REST_PER_PAGE, "page": page}
        data = client.get_json(url, params=params)
        if not data:
            break
        rels = [
            r for r in data
            if since <= (r.get("published_at") or r.get("created_at") or "") <= until
        ]
        _write_page(ctx.raw_releases_dir, page, rels)
        total += len(rels)
        checkpoint.update("releases", page=page, count=total)
        if progress:
            progress(f"releases: fetched {total}")
        if len(data) < config.GITHUB_REST_PER_PAGE:
            break
        page += 1

    checkpoint.mark_done("releases", count=total)
    return total
