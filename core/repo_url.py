"""Parse and validate GitHub repository references.

Accepts every common form a user might paste:
  * https://github.com/owner/name
  * https://github.com/owner/name.git
  * https://github.com/owner/name/tree/main/some/path
  * http://www.github.com/owner/name/pull/12
  * git@github.com:owner/name.git            (SSH)
  * ssh://git@github.com/owner/name.git
  * owner/name                               (shorthand)

Produces a normalized :class:`RepoRef`. Rejects non-GitHub hosts and
malformed input with a clear :class:`InvalidRepoURL`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse


class InvalidRepoURL(ValueError):
    """Raised when a string cannot be parsed as a GitHub repo reference."""


# GitHub owner and repo naming rules (practical subset):
#   owner: alphanumeric or hyphen, 1-39 chars, no leading/trailing hyphen
#   repo:  alphanumeric, hyphen, underscore, dot; 1-100 chars
_OWNER_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$")
_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")

_ALLOWED_HOSTS = {"github.com", "www.github.com"}


@dataclass(frozen=True)
class RepoRef:
    """A validated, normalized reference to a GitHub repository."""

    owner: str
    name: str

    @property
    def slug(self) -> str:
        """Filesystem-safe identifier: ``owner_name`` (lowercased)."""
        return f"{self.owner}_{self.name}".lower()

    @property
    def full_name(self) -> str:
        """Canonical ``owner/name`` GitHub identifier (original case)."""
        return f"{self.owner}/{self.name}"

    @property
    def html_url(self) -> str:
        return f"https://github.com/{self.owner}/{self.name}"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.full_name


def _clean_name(name: str) -> str:
    """Strip a trailing ``.git`` from a repo name."""
    if name.endswith(".git"):
        name = name[: -len(".git")]
    return name


def _validate_parts(owner: str, name: str, raw: str) -> RepoRef:
    owner = owner.strip()
    name = _clean_name(name.strip())
    if not owner or not name:
        raise InvalidRepoURL(f"Could not extract owner/name from {raw!r}")
    if not _OWNER_RE.match(owner):
        raise InvalidRepoURL(f"Invalid GitHub owner {owner!r} in {raw!r}")
    if not _REPO_RE.match(name) or name in {".", ".."}:
        raise InvalidRepoURL(f"Invalid GitHub repo name {name!r} in {raw!r}")
    return RepoRef(owner=owner, name=name)


def parse_repo_url(raw: str) -> RepoRef:
    """Parse any supported GitHub reference form into a :class:`RepoRef`.

    Raises :class:`InvalidRepoURL` on anything unparseable or non-GitHub.
    """
    if raw is None or not str(raw).strip():
        raise InvalidRepoURL("Empty repository reference")
    text = str(raw).strip()

    # 1) SSH scp-like form: git@github.com:owner/name(.git)
    m = re.match(r"^(?:ssh://)?git@([^:/]+)[:/](?P<path>.+)$", text)
    if m:
        host = m.group(1).lower()
        if host not in _ALLOWED_HOSTS:
            raise InvalidRepoURL(f"Not a github.com host: {host!r}")
        parts = m.group("path").strip("/").split("/")
        if len(parts) < 2:
            raise InvalidRepoURL(f"Could not extract owner/name from {raw!r}")
        return _validate_parts(parts[0], parts[1], raw)

    # 2) Full URL with a scheme (http/https/git)
    if "://" in text:
        parsed = urlparse(text)
        if parsed.scheme.lower() not in {"http", "https", "git"}:
            raise InvalidRepoURL(
                f"Unsupported URL scheme {parsed.scheme!r} in {raw!r}"
            )
        host = (parsed.hostname or "").lower()
        if host not in _ALLOWED_HOSTS:
            raise InvalidRepoURL(f"Not a github.com URL: {raw!r}")
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) < 2:
            raise InvalidRepoURL(f"URL missing owner/name: {raw!r}")
        return _validate_parts(parts[0], parts[1], raw)

    # 3) Bare github.com/owner/name (no scheme)
    if text.lower().startswith(("github.com/", "www.github.com/")):
        parts = [p for p in text.split("/") if p]
        # parts[0] is the host
        if len(parts) < 3:
            raise InvalidRepoURL(f"Could not extract owner/name from {raw!r}")
        return _validate_parts(parts[1], parts[2], raw)

    # 4) Shorthand owner/name (exactly, optionally with .git / trailing slash)
    parts = [p for p in text.split("/") if p]
    if len(parts) == 2:
        return _validate_parts(parts[0], parts[1], raw)

    raise InvalidRepoURL(f"Unrecognized GitHub repository reference: {raw!r}")
