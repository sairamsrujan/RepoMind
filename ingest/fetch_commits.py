"""Fetch commits via the GitHub REST API, scoped to a date window.

Uses explicit page numbers (not Link-following) so an interrupted run resumes
from the next page. Each page is written to its own file
``raw/commits/page_NNNN.json`` — a partial page from a killed process is simply
overwritten on resume, avoiding corrupt state.
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


def fetch_commits(
    client: GitHubClient,
    ctx: RepositoryContext,
    ref: RepoRef,
    since: str,
    until: str,
    checkpoint: Checkpoint,
    progress: ProgressFn | None = None,
) -> int:
    """Fetch commits in [since, until]; return total count fetched.

    ``since``/``until`` are ISO-8601 timestamps (commit author date filter).
    """
    if checkpoint.is_done("commits"):
        return checkpoint.get("commits")["count"]

    url = f"{config.GITHUB_API_URL}/repos/{ref.owner}/{ref.name}/commits"
    state = checkpoint.get("commits")
    start_page = state["page"] + 1
    total = state["count"]

    page = start_page
    while True:
        params = {
            "since": since,
            "until": until,
            "per_page": config.GITHUB_REST_PER_PAGE,
            "page": page,
        }
        data = client.get_json(url, params=params)
        if not data:
            break
        _write_page(ctx.raw_commits_dir, page, data)
        total += len(data)
        checkpoint.update("commits", page=page, count=total)
        if progress:
            progress(f"commits: fetched {total}")
        if len(data) < config.GITHUB_REST_PER_PAGE:
            break
        page += 1

    checkpoint.mark_done("commits", count=total)
    return total
