"""Fetch issues via the GitHub REST API, scoped to a date window.

The REST ``/issues`` endpoint returns pull requests too; those are filtered out
here (PRs are fetched separately, richly, via GraphQL). ``since`` filters by
updated-at; the ``until`` bound is applied client-side on created-at.
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


def fetch_issues(
    client: GitHubClient,
    ctx: RepositoryContext,
    ref: RepoRef,
    since: str,
    until: str,
    checkpoint: Checkpoint,
    progress: ProgressFn | None = None,
) -> int:
    """Fetch issues (excluding PRs) updated at/after ``since``; return count."""
    if checkpoint.is_done("issues"):
        return checkpoint.get("issues")["count"]

    url = f"{config.GITHUB_API_URL}/repos/{ref.owner}/{ref.name}/issues"
    state = checkpoint.get("issues")
    start_page = state["page"] + 1
    total = state["count"]

    page = start_page
    while True:
        params = {
            "state": "all",
            "since": since,
            "sort": "created",
            "direction": "asc",
            "per_page": config.GITHUB_REST_PER_PAGE,
            "page": page,
        }
        data = client.get_json(url, params=params)
        if not data:
            break
        # Drop pull requests (they carry a "pull_request" key) and out-of-window.
        issues = [
            it for it in data
            if "pull_request" not in it and it.get("created_at", "") <= until
        ]
        _write_page(ctx.raw_issues_dir, page, issues)
        total += len(issues)
        checkpoint.update("issues", page=page, count=total)
        if progress:
            progress(f"issues: fetched {total}")
        if len(data) < config.GITHUB_REST_PER_PAGE:
            break
        page += 1

    checkpoint.mark_done("issues", count=total)
    return total
