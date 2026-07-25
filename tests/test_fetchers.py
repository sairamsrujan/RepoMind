"""Tests for the fetchers: pagination, date-window filtering, and resume."""
from __future__ import annotations

import json

import pytest

from core.context import RepositoryContext
from core.repo_url import parse_repo_url
from ingest.checkpoint import Checkpoint
from ingest.fetch_commits import fetch_commits
from ingest.fetch_prs import fetch_prs


class MockRESTClient:
    """Serves commit pages keyed by the ``page`` param; records pages asked."""

    def __init__(self, pages):
        # pages: list of page payloads (index 0 -> page 1, etc.)
        self._pages = pages
        self.pages_requested = []
        self.fail_on_page = None

    def get_json(self, url, params=None):
        page = params["page"]
        self.pages_requested.append(page)
        if self.fail_on_page is not None and page == self.fail_on_page:
            raise RuntimeError("simulated interruption")
        if 1 <= page <= len(self._pages):
            return self._pages[page - 1]
        return []


def _ctx(tmp_path, slug="owner/repo"):
    ref = parse_repo_url(slug)
    ctx = RepositoryContext.for_ref(ref, tmp_path / "repositories")
    ctx.ensure_dirs()
    return ref, ctx


def _full_page(n, since_date="2024-06-15T00:00:00Z"):
    return [{"sha": f"sha{n}_{i}", "commit": {"author": {"date": since_date}}}
            for i in range(100)]


def test_fetch_commits_paginates(tmp_path):
    ref, ctx = _ctx(tmp_path)
    cp = Checkpoint(ctx.checkpoint_path)
    # Two full pages (100 each) then a short page (30) -> stops.
    client = MockRESTClient([_full_page(1), _full_page(2), _full_page(3)[:30]])
    total = fetch_commits(client, ctx, ref, "2024-01-01T00:00:00Z",
                          "2024-12-31T00:00:00Z", cp)
    assert total == 230
    assert cp.is_done("commits")
    # Page files written
    files = sorted(ctx.raw_commits_dir.glob("page_*.json"))
    assert len(files) == 3


def test_fetch_commits_resume_after_interruption(tmp_path):
    ref, ctx = _ctx(tmp_path)
    cp = Checkpoint(ctx.checkpoint_path)
    # Interrupt while fetching page 2.
    client = MockRESTClient([_full_page(1), _full_page(2), _full_page(3)[:30]])
    client.fail_on_page = 2
    with pytest.raises(RuntimeError):
        fetch_commits(client, ctx, ref, "2024-01-01T00:00:00Z",
                      "2024-12-31T00:00:00Z", cp)
    # Page 1 committed, page 2 failed -> checkpoint at page 1, not done.
    assert cp.get("commits")["page"] == 1
    assert not cp.is_done("commits")

    # Re-run with a fresh client -> must resume at page 2, not refetch page 1.
    cp2 = Checkpoint(ctx.checkpoint_path)  # reload from disk
    client2 = MockRESTClient([_full_page(1), _full_page(2), _full_page(3)[:30]])
    total = fetch_commits(client2, ctx, ref, "2024-01-01T00:00:00Z",
                          "2024-12-31T00:00:00Z", cp2)
    assert total == 230
    assert 1 not in client2.pages_requested   # page 1 NOT refetched
    assert 2 in client2.pages_requested
    assert cp2.is_done("commits")


def test_fetch_commits_skips_if_done(tmp_path):
    ref, ctx = _ctx(tmp_path)
    cp = Checkpoint(ctx.checkpoint_path)
    cp.mark_done("commits", count=17)
    client = MockRESTClient([_full_page(1)])
    total = fetch_commits(client, ctx, ref, "2024-01-01T00:00:00Z",
                          "2024-12-31T00:00:00Z", cp)
    assert total == 17
    assert client.pages_requested == []  # nothing fetched


# ---- PR / GraphQL -------------------------------------------------------- #

class MockGraphQLClient:
    def __init__(self, pages):
        # pages: list of (nodes, hasNextPage, endCursor)
        self._pages = pages
        self._idx_by_cursor = {}
        self.cursors_seen = []

    def graphql(self, query, variables):
        cursor = variables["cursor"]
        self.cursors_seen.append(cursor)
        idx = 0 if cursor is None else int(cursor.split(":")[1])
        nodes, has_next, end_cursor = self._pages[idx]
        return {
            "repository": {
                "pullRequests": {
                    "pageInfo": {"hasNextPage": has_next, "endCursor": end_cursor},
                    "nodes": nodes,
                }
            }
        }


def _pr(number, created):
    return {"number": number, "title": f"PR {number}", "body": "",
            "state": "MERGED", "url": f"https://github.com/o/r/pull/{number}",
            "createdAt": created, "mergedAt": created, "closedAt": created,
            "author": {"login": "dev"}, "mergeCommit": {"oid": f"m{number}"},
            "labels": {"nodes": []},
            "commits": {"totalCount": 1, "nodes": [{"commit": {"oid": f"c{number}",
                        "messageHeadline": "x"}}]},
            "reviews": {"nodes": []},
            "closingIssuesReferences": {"nodes": []}}


def test_fetch_prs_window_and_stop(tmp_path):
    ref, ctx = _ctx(tmp_path)
    cp = Checkpoint(ctx.checkpoint_path)
    # Page 0: two in-window PRs. Page 1: one in-window + one older-than-since.
    pages = [
        ([_pr(10, "2024-06-01T00:00:00Z"), _pr(9, "2024-05-01T00:00:00Z")],
         True, "cur:1"),
        ([_pr(8, "2024-02-01T00:00:00Z"), _pr(7, "2023-11-01T00:00:00Z")],
         True, "cur:2"),
    ]
    client = MockGraphQLClient(pages)
    total = fetch_prs(client, ctx, ref, "2024-01-01T00:00:00Z",
                      "2024-12-31T00:00:00Z", cp)
    # PR 7 (2023-11) is out of window -> excluded; and paging stops there.
    assert total == 3
    assert cp.is_done("prs")
    batches = sorted(ctx.raw_prs_dir.glob("batch_*.json"))
    assert len(batches) == 2
    # Verify only in-window PRs persisted.
    saved_numbers = set()
    for b in batches:
        for pr in json.loads(b.read_text()):
            saved_numbers.add(pr["number"])
    assert saved_numbers == {10, 9, 8}
