"""Thin GitHub REST + GraphQL client with auth, backoff, and ETag caching.

Design goals:
  * Read ``GITHUB_TOKEN`` from config/.env; work unauthenticated too (limited).
  * Exponential backoff honoring ``Retry-After`` on 403/429.
  * Raise a clear, typed error on 404/451 (private/missing/legal-blocked repo).
  * Conditional requests via ETag caching to save rate-limit budget.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import requests

import config


class GitHubError(RuntimeError):
    """Base class for GitHub client errors."""


class RepoNotFound(GitHubError):
    """404/451 — repository is missing, private, or legally blocked."""


class RateLimitExceeded(GitHubError):
    """Rate limit could not be recovered from within MAX_RETRIES."""


class GraphQLError(GitHubError):
    """GraphQL endpoint returned an ``errors`` array."""


@dataclass
class RateLimitStatus:
    remaining: int | None = None
    limit: int | None = None
    reset_epoch: int | None = None


class _ETagCache:
    """Persistent map: cache-key -> {etag, data}. Used for REST GETs."""

    def __init__(self, path: Path | None):
        self.path = Path(path) if path else None
        self._data: dict[str, Any] = {}
        if self.path and self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def get(self, key: str) -> dict[str, Any] | None:
        return self._data.get(key)

    def put(self, key: str, etag: str, data: Any) -> None:
        self._data[key] = {"etag": etag, "data": data}

    def flush(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data), encoding="utf-8")


class GitHubClient:
    """A minimal, rate-limit-aware GitHub API client."""

    def __init__(
        self,
        token: str | None = None,
        etag_cache_path: str | Path | None = None,
        session: requests.Session | None = None,
    ):
        self.token = token if token is not None else config.GITHUB_TOKEN
        self.session = session or requests.Session()
        self._cache = _ETagCache(Path(etag_cache_path) if etag_cache_path else None)
        self.rate = RateLimitStatus()

    # ------------------------------------------------------------------ #
    # Headers
    # ------------------------------------------------------------------ #
    def _headers(self, accept: str) -> dict[str, str]:
        headers = {
            "Accept": accept,
            "User-Agent": "RepoMind/1.0",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _update_rate(self, resp: requests.Response) -> None:
        def _int(h: str) -> int | None:
            v = resp.headers.get(h)
            return int(v) if v is not None and v.isdigit() else None

        self.rate = RateLimitStatus(
            remaining=_int("X-RateLimit-Remaining"),
            limit=_int("X-RateLimit-Limit"),
            reset_epoch=_int("X-RateLimit-Reset"),
        )

    # ------------------------------------------------------------------ #
    # Core request with backoff
    # ------------------------------------------------------------------ #
    def _rate_limit_wait(self, resp: requests.Response, attempt: int) -> float:
        """Seconds to wait before retrying a rate-limited request.

        Returns a negative number to signal "do not retry — the reset is too
        far away", so callers fail fast instead of sleeping through futile
        retries (e.g. an exhausted unauthenticated primary limit).
        """
        retry_after = resp.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            # Secondary rate limit — honor it if reasonably short.
            wait = int(retry_after)
            return float(wait) if wait <= config.MAX_RATE_WAIT_SECONDS else -1.0
        remaining = resp.headers.get("X-RateLimit-Remaining")
        reset = resp.headers.get("X-RateLimit-Reset")
        if remaining == "0" and reset and reset.isdigit():
            wait = int(reset) - int(time.time())
            if wait > config.MAX_RATE_WAIT_SECONDS:
                return -1.0  # reset too far off; don't retry
            return float(max(1, wait))
        # Unknown/secondary throttle without headers: short exponential backoff.
        return config.BACKOFF_BASE_SECONDS * (2 ** attempt)

    def _request(
        self,
        method: str,
        url: str,
        *,
        accept: str = "application/vnd.github+json",
        params: dict | None = None,
        json_body: dict | None = None,
        extra_headers: dict | None = None,
    ) -> requests.Response:
        headers = self._headers(accept)
        if extra_headers:
            headers.update(extra_headers)

        last_exc: Exception | None = None
        for attempt in range(config.MAX_RETRIES):
            try:
                resp = self.session.request(
                    method,
                    url,
                    headers=headers,
                    params=params,
                    json=json_body,
                    timeout=config.HTTP_TIMEOUT_SECONDS,
                )
            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(config.BACKOFF_BASE_SECONDS * (2 ** attempt))
                continue

            self._update_rate(resp)

            if resp.status_code in (404, 451):
                raise RepoNotFound(
                    f"{resp.status_code} for {url} — repository is missing, "
                    f"private, or unavailable."
                )

            if resp.status_code in (403, 429):
                # Distinguish rate limiting from a genuine auth 403.
                is_rate = (
                    resp.headers.get("X-RateLimit-Remaining") == "0"
                    or "Retry-After" in resp.headers
                    or "rate limit" in resp.text.lower()
                )
                if not is_rate:
                    raise GitHubError(f"403 Forbidden for {url}: {resp.text[:200]}")
                wait = self._rate_limit_wait(resp, attempt)
                reset = resp.headers.get("X-RateLimit-Reset", "")
                if wait < 0 or attempt >= config.MAX_RETRIES - 1:
                    hint = (
                        " Set GITHUB_TOKEN in .env for a much higher limit."
                        if not self.token else ""
                    )
                    raise RateLimitExceeded(
                        f"GitHub rate limit exceeded for {url} "
                        f"(reset epoch {reset}).{hint}"
                    )
                time.sleep(wait)
                continue

            if resp.status_code >= 500 and attempt < config.MAX_RETRIES - 1:
                time.sleep(config.BACKOFF_BASE_SECONDS * (2 ** attempt))
                continue

            return resp

        raise GitHubError(f"Request to {url} failed after retries: {last_exc}")

    # ------------------------------------------------------------------ #
    # REST
    # ------------------------------------------------------------------ #
    def get_json(self, url: str, params: dict | None = None) -> Any:
        """Single REST GET returning parsed JSON (no pagination)."""
        resp = self._request("GET", url, params=params)
        if resp.status_code >= 400:
            raise GitHubError(f"{resp.status_code} for {url}: {resp.text[:200]}")
        return resp.json()

    def get_repo(self, owner: str, name: str) -> dict[str, Any]:
        """Fetch repository metadata (validates existence)."""
        return self.get_json(f"{config.GITHUB_API_URL}/repos/{owner}/{name}")

    def paginate(
        self, url: str, params: dict | None = None, cache_key_prefix: str = ""
    ) -> Iterator[dict[str, Any]]:
        """Yield items across all pages of a REST list endpoint.

        Follows the ``Link: rel="next"`` header. Uses ETag caching per page.
        """
        params = dict(params or {})
        params.setdefault("per_page", config.GITHUB_REST_PER_PAGE)
        page = 1
        next_url: str | None = url
        while next_url:
            page_params = params if next_url == url else None
            cache_key = f"{cache_key_prefix}:{next_url}:{json.dumps(page_params, sort_keys=True)}"
            cached = self._cache.get(cache_key)
            extra = {"If-None-Match": cached["etag"]} if cached else None

            resp = self._request(
                "GET", next_url, params=page_params, extra_headers=extra
            )

            if resp.status_code == 304 and cached:
                items = cached["data"]
            else:
                if resp.status_code >= 400:
                    raise GitHubError(
                        f"{resp.status_code} for {next_url}: {resp.text[:200]}"
                    )
                items = resp.json()
                etag = resp.headers.get("ETag")
                if etag:
                    self._cache.put(cache_key, etag, items)

            if isinstance(items, list):
                yield from items
            else:  # defensive: some endpoints wrap results
                yield items

            next_url = self._next_link(resp)
            page += 1

        self._cache.flush()

    @staticmethod
    def _next_link(resp: requests.Response) -> str | None:
        link = resp.headers.get("Link")
        if not link:
            return None
        for part in link.split(","):
            segments = part.split(";")
            if len(segments) < 2:
                continue
            url_part = segments[0].strip().strip("<>")
            rel_part = segments[1].strip()
            if rel_part == 'rel="next"':
                return url_part
        return None

    # ------------------------------------------------------------------ #
    # GraphQL
    # ------------------------------------------------------------------ #
    def graphql(self, query: str, variables: dict | None = None) -> dict[str, Any]:
        """Execute a GraphQL query and return the ``data`` object."""
        resp = self._request(
            "POST",
            config.GITHUB_GRAPHQL_URL,
            json_body={"query": query, "variables": variables or {}},
        )
        if resp.status_code >= 400:
            raise GitHubError(f"GraphQL {resp.status_code}: {resp.text[:300]}")
        payload = resp.json()
        if payload.get("errors"):
            # NOT_FOUND type surfaces as a RepoNotFound for callers.
            for err in payload["errors"]:
                if err.get("type") == "NOT_FOUND":
                    raise RepoNotFound(f"GraphQL NOT_FOUND: {err.get('message')}")
            raise GraphQLError(f"GraphQL errors: {payload['errors']}")
        return payload.get("data", {})

    def flush_cache(self) -> None:
        self._cache.flush()
