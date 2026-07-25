"""Fetch pull requests via the GitHub GraphQL API.

A single query pulls, per PR: metadata, author, merge commit, up to 50 commit
SHAs, reviews, linked/closing issues, and labels — far fewer round-trips than
the equivalent REST calls. Results are cursor-paginated (newest first) and the
cursor is checkpointed so an interrupted run resumes mid-stream.
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

_PR_QUERY = """
query($owner:String!, $name:String!, $pageSize:Int!, $cursor:String) {
  repository(owner:$owner, name:$name) {
    pullRequests(first:$pageSize, after:$cursor,
                 orderBy:{field:CREATED_AT, direction:DESC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number title body state url
        createdAt mergedAt closedAt
        author { login }
        mergeCommit { oid }
        labels(first:15) { nodes { name } }
        commits(first:50) {
          totalCount
          nodes { commit { oid messageHeadline } }
        }
        reviews(first:30) {
          nodes { author { login } state body submittedAt url }
        }
        closingIssuesReferences(first:15) {
          nodes { number title url }
        }
      }
    }
  }
}
"""


def _write_batch(directory, batch: int, nodes: list) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"batch_{batch:04d}.json").write_text(
        json.dumps(nodes), encoding="utf-8"
    )


def fetch_prs(
    client: GitHubClient,
    ctx: RepositoryContext,
    ref: RepoRef,
    since: str,
    until: str,
    checkpoint: Checkpoint,
    progress: ProgressFn | None = None,
) -> int:
    """Fetch PRs created within [since, until]; return count.

    PRs are returned newest-first, so we stop once a page's PRs predate
    ``since``.
    """
    if checkpoint.is_done("prs"):
        return checkpoint.get("prs")["count"]

    state = checkpoint.get("prs")
    cursor = state["cursor"]
    batch = state["page"]
    total = state["count"]

    while True:
        variables = {
            "owner": ref.owner,
            "name": ref.name,
            "pageSize": config.GITHUB_GRAPHQL_PAGE_SIZE,
            "cursor": cursor,
        }
        data = client.graphql(_PR_QUERY, variables)
        prs = data["repository"]["pullRequests"]
        nodes = prs["nodes"]
        page_info = prs["pageInfo"]

        in_window = [
            n for n in nodes if since <= (n.get("createdAt") or "") <= until
        ]
        older_than_since = [
            n for n in nodes if (n.get("createdAt") or "") < since
        ]

        if in_window:
            batch += 1
            _write_batch(ctx.raw_prs_dir, batch, in_window)
            total += len(in_window)
            checkpoint.update(
                "prs", page=batch, count=total, cursor=page_info["endCursor"]
            )
            if progress:
                progress(f"prs: fetched {total}")

        # Stop if we've paged past the window (some nodes older than `since`)
        # or there are no more pages.
        if older_than_since or not page_info["hasNextPage"]:
            break
        cursor = page_info["endCursor"]
        checkpoint.update("prs", cursor=cursor)

    checkpoint.mark_done("prs", count=total)
    return total
