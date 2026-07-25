"""Shared pytest fixtures and path setup.

Ensures the project root is importable so `import config`, `import core...`
etc. work regardless of where pytest is invoked from.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def fixture_repo(tmp_path):
    """A RepositoryContext populated with raw data in real GitHub API shapes.

    Encodes a known ground-truth link: PR #101 and commit ``c0ffee1`` both
    close issue #1 ("fix the crash"), and PR #101 shipped in release v1.2.0.
    """
    from core.context import RepositoryContext
    from core.repo_url import parse_repo_url

    ref = parse_repo_url("acme/widgets")
    ctx = RepositoryContext.for_ref(ref, tmp_path / "repositories")
    ctx.ensure_dirs()

    # --- commits (REST list shape) --- #
    commits = [
        {
            "sha": "c0ffee1000000000000000000000000000000000",
            "html_url": "https://github.com/acme/widgets/commit/c0ffee1",
            "author": {"login": "alice"},
            "commit": {
                "message": "Fix null pointer crash on startup\n\nfixes #1",
                "author": {"name": "Alice", "date": "2024-03-02T10:00:00Z"},
            },
        },
        {
            "sha": "deadbee2000000000000000000000000000000000",
            "html_url": "https://github.com/acme/widgets/commit/deadbee2",
            "author": {"login": "bob"},
            "commit": {
                "message": "Add caching layer for widget lookups",
                "author": {"name": "Bob", "date": "2024-03-05T12:00:00Z"},
            },
        },
    ]
    _write_json(ctx.raw_commits_dir / "page_0001.json", commits)

    # --- pull requests (GraphQL nodes shape) --- #
    prs = [
        {
            "number": 101,
            "title": "Fix startup crash",
            "body": "This resolves the crash reported in #1 by null-checking config.",
            "state": "MERGED",
            "url": "https://github.com/acme/widgets/pull/101",
            "createdAt": "2024-03-02T09:00:00Z",
            "mergedAt": "2024-03-03T09:00:00Z",
            "closedAt": "2024-03-03T09:00:00Z",
            "author": {"login": "alice"},
            "mergeCommit": {"oid": "c0ffee1000000000000000000000000000000000"},
            "labels": {"nodes": [{"name": "bug"}]},
            "commits": {"totalCount": 1, "nodes": [
                {"commit": {"oid": "c0ffee1000000000000000000000000000000000",
                            "messageHeadline": "Fix null pointer crash"}}]},
            "reviews": {"nodes": [
                {"author": {"login": "bob"}, "state": "APPROVED",
                 "body": "Looks good, nice catch on the null check.",
                 "submittedAt": "2024-03-02T18:00:00Z",
                 "url": "https://github.com/acme/widgets/pull/101#r1"}]},
            "closingIssuesReferences": {"nodes": [
                {"number": 1, "title": "Crash on startup",
                 "url": "https://github.com/acme/widgets/issues/1"}]},
        },
        {
            "number": 102,
            "title": "Add caching layer",
            "body": "Speeds up widget lookups. See #2 for the discussion.",
            "state": "MERGED",
            "url": "https://github.com/acme/widgets/pull/102",
            "createdAt": "2024-03-05T09:00:00Z",
            "mergedAt": "2024-03-06T09:00:00Z",
            "closedAt": "2024-03-06T09:00:00Z",
            "author": {"login": "bob"},
            "mergeCommit": {"oid": "deadbee2000000000000000000000000000000000"},
            "labels": {"nodes": [{"name": "enhancement"}]},
            "commits": {"totalCount": 1, "nodes": [
                {"commit": {"oid": "deadbee2000000000000000000000000000000000",
                            "messageHeadline": "Add caching layer"}}]},
            "reviews": {"nodes": []},
            "closingIssuesReferences": {"nodes": []},
        },
    ]
    _write_json(ctx.raw_prs_dir / "batch_0001.json", prs)

    # --- issues (REST shape) --- #
    issues = [
        {"number": 1, "title": "Crash on startup",
         "body": "The app crashes with a null pointer when config is missing.",
         "state": "closed", "user": {"login": "carol"},
         "created_at": "2024-03-01T08:00:00Z",
         "html_url": "https://github.com/acme/widgets/issues/1"},
        {"number": 2, "title": "Widget lookups are slow",
         "body": "Looking up widgets takes too long under load. Can we cache?",
         "state": "open", "user": {"login": "dave"},
         "created_at": "2024-03-04T08:00:00Z",
         "html_url": "https://github.com/acme/widgets/issues/2"},
    ]
    _write_json(ctx.raw_issues_dir / "page_0001.json", issues)

    # --- releases (REST shape) --- #
    releases = [
        {"tag_name": "v1.2.0", "name": "v1.2.0 - Stability",
         "body": "Fixes the startup crash and adds a caching layer.",
         "author": {"login": "alice"},
         "published_at": "2024-03-10T00:00:00Z",
         "html_url": "https://github.com/acme/widgets/releases/tag/v1.2.0"},
    ]
    _write_json(ctx.raw_releases_dir / "page_0001.json", releases)

    return ctx
