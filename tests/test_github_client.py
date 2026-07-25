"""Tests for the GitHub client: backoff, Retry-After, 404/451, ETag caching."""
from __future__ import annotations

import json

import pytest

from ingest.github_client import (
    GitHubClient,
    RateLimitExceeded,
    RepoNotFound,
)


class FakeResponse:
    def __init__(self, status_code, headers=None, body=None, text=""):
        self.status_code = status_code
        self.headers = headers or {}
        self._body = body
        self.text = text or (json.dumps(body) if body is not None else "")

    def json(self):
        return self._body


class FakeSession:
    """Returns queued responses in order; records requests made."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def request(self, method, url, headers=None, params=None, json=None, timeout=None):
        self.calls.append({"method": method, "url": url, "params": params,
                           "headers": headers})
        return self._responses.pop(0)


def test_404_raises_repo_not_found(monkeypatch):
    session = FakeSession([FakeResponse(404, text="Not Found")])
    client = GitHubClient(token="x", session=session)
    with pytest.raises(RepoNotFound):
        client.get_json("https://api.github.com/repos/a/b")


def test_451_raises_repo_not_found():
    session = FakeSession([FakeResponse(451, text="Unavailable For Legal Reasons")])
    client = GitHubClient(token="x", session=session)
    with pytest.raises(RepoNotFound):
        client.get_repo("a", "b")


def test_retry_after_then_success(monkeypatch):
    # First a 403 rate-limit with Retry-After, then a 200.
    sleeps = []
    monkeypatch.setattr("ingest.github_client.time.sleep", lambda s: sleeps.append(s))
    session = FakeSession([
        FakeResponse(403, headers={"Retry-After": "3", "X-RateLimit-Remaining": "0"},
                     text="rate limit exceeded"),
        FakeResponse(200, body=[{"sha": "abc"}]),
    ])
    client = GitHubClient(token="x", session=session)
    data = client.get_json("https://api.github.com/repos/a/b/commits")
    assert data == [{"sha": "abc"}]
    assert sleeps == [3.0]  # honored Retry-After


def test_rate_limit_exhausts_retries(monkeypatch):
    monkeypatch.setattr("ingest.github_client.time.sleep", lambda s: None)
    # Always 429 rate-limited -> should raise after MAX_RETRIES.
    responses = [
        FakeResponse(429, headers={"X-RateLimit-Remaining": "0"},
                     text="rate limit") for _ in range(10)
    ]
    session = FakeSession(responses)
    client = GitHubClient(token="x", session=session)
    with pytest.raises(RateLimitExceeded):
        client.get_json("https://api.github.com/x")


def test_rate_limit_far_reset_fails_fast(monkeypatch):
    # remaining=0 and reset ~1h away -> must NOT sleep-retry; raise immediately.
    slept = []
    monkeypatch.setattr("ingest.github_client.time.sleep", lambda s: slept.append(s))
    far_reset = str(int(__import__("time").time()) + 3600)
    session = FakeSession([
        FakeResponse(403, headers={"X-RateLimit-Remaining": "0",
                                   "X-RateLimit-Reset": far_reset},
                     text="API rate limit exceeded"),
    ])
    client = GitHubClient(token=None, session=session)
    with pytest.raises(RateLimitExceeded):
        client.get_json("https://api.github.com/graphql")
    assert slept == []          # failed fast, no pointless long sleeps
    assert len(session.calls) == 1


def test_etag_cache_304_returns_cached(tmp_path):
    cache_path = tmp_path / "etags.json"
    # First call: 200 with ETag; second call: 304 -> serve cached.
    session = FakeSession([
        FakeResponse(200, headers={"ETag": 'W/"abc"'}, body=[{"n": 1}]),
        FakeResponse(304, headers={"ETag": 'W/"abc"'}),
    ])
    client = GitHubClient(token="x", session=session, etag_cache_path=cache_path)
    url = "https://api.github.com/repos/a/b/issues"
    first = list(client.paginate(url))
    assert first == [{"n": 1}]

    # New client, same cache file -> conditional request served from cache.
    client2 = GitHubClient(token="x", session=session, etag_cache_path=cache_path)
    second = list(client2.paginate(url))
    assert second == [{"n": 1}]
    # The second session response was a 304; verify If-None-Match was sent.
    assert session.calls[-1]["headers"].get("If-None-Match") == 'W/"abc"'


def test_graphql_not_found_raises():
    session = FakeSession([
        FakeResponse(200, body={"errors": [{"type": "NOT_FOUND", "message": "x"}]}),
    ])
    client = GitHubClient(token="x", session=session)
    with pytest.raises(RepoNotFound):
        client.graphql("query{}", {})
