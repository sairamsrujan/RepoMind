"""Tests for core.repo_url — every accepted form and malformed input."""
from __future__ import annotations

import pytest

from core.repo_url import InvalidRepoURL, RepoRef, parse_repo_url


@pytest.mark.parametrize(
    "raw,owner,name",
    [
        ("https://github.com/pallets/flask", "pallets", "flask"),
        ("https://github.com/pallets/flask.git", "pallets", "flask"),
        ("https://github.com/pallets/flask/", "pallets", "flask"),
        ("https://github.com/pallets/flask/tree/main/src", "pallets", "flask"),
        ("http://www.github.com/pallets/flask/pull/12", "pallets", "flask"),
        ("github.com/pallets/flask", "pallets", "flask"),
        ("www.github.com/pallets/flask", "pallets", "flask"),
        ("git@github.com:pallets/flask.git", "pallets", "flask"),
        ("git@github.com:pallets/flask", "pallets", "flask"),
        ("ssh://git@github.com/pallets/flask.git", "pallets", "flask"),
        ("pallets/flask", "pallets", "flask"),
        ("  pallets/flask  ", "pallets", "flask"),
        ("https://github.com/psf/requests-html", "psf", "requests-html"),
        ("owner-1/repo.name_2", "owner-1", "repo.name_2"),
    ],
)
def test_parse_valid_forms(raw, owner, name):
    ref = parse_repo_url(raw)
    assert isinstance(ref, RepoRef)
    assert ref.owner == owner
    assert ref.name == name


def test_slug_and_urls():
    ref = parse_repo_url("https://github.com/Pallets/Flask")
    assert ref.slug == "pallets_flask"          # lowercased, underscore
    assert ref.full_name == "Pallets/Flask"     # original case preserved
    assert ref.html_url == "https://github.com/Pallets/Flask"


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        None,
        "not a url",
        "https://gitlab.com/owner/name",       # wrong host
        "https://github.com/onlyowner",         # missing name
        "git@bitbucket.org:owner/name.git",     # wrong host
        "https://example.com/pallets/flask",    # wrong host
        "ftp://github.com/owner/name",          # non github host resolves empty
        "owner//",                               # empty name
        "-bad/name",                             # invalid owner (leading hyphen)
    ],
)
def test_parse_rejects_malformed(raw):
    with pytest.raises(InvalidRepoURL):
        parse_repo_url(raw)


def test_renamed_repo_produces_new_slug():
    """A renamed repo yields a distinct slug -> distinct on-disk directory."""
    old = parse_repo_url("https://github.com/owner/old-name")
    new = parse_repo_url("https://github.com/owner/new-name")
    assert old.slug != new.slug
