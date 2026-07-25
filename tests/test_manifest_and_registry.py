"""Tests for manifest read/write/validate, the reuse rule, and the registry."""
from __future__ import annotations

import json

import pytest

import config
from core import manifest as M
from core import registry
from core.context import RepositoryContext
from core.repo_url import parse_repo_url


def _ready_manifest(ref):
    m = M.new_manifest(ref)
    m["status"] = "ready"
    m["pipeline"]["embedding_dim"] = 1024
    m["stats"]["chunks"] = 42
    return m


def test_new_manifest_validates():
    ref = parse_repo_url("pallets/flask")
    m = M.new_manifest(ref)
    M.validate_manifest(m)  # should not raise
    assert m["status"] == "pending"
    assert m["repo"]["slug"] == "pallets_flask"
    assert m["pipeline"]["embedding_model"] == config.EMBEDDING_MODEL


def test_write_then_read_roundtrip(tmp_path):
    ref = parse_repo_url("pallets/flask")
    m = _ready_manifest(ref)
    path = tmp_path / "manifest.json"
    M.write_manifest(path, m)
    loaded = M.read_manifest(path)
    assert loaded["repo"]["owner"] == "pallets"
    assert loaded["status"] == "ready"
    # updated_at refreshed on write
    assert loaded["updated_at"]


def test_read_missing_raises(tmp_path):
    with pytest.raises(M.ManifestError):
        M.read_manifest(tmp_path / "nope.json")


def test_read_corrupt_json_raises(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text("{ not json", encoding="utf-8")
    with pytest.raises(M.ManifestError):
        M.read_manifest(p)


def test_validate_rejects_bad_status():
    ref = parse_repo_url("pallets/flask")
    m = M.new_manifest(ref)
    m["status"] = "bogus"
    with pytest.raises(M.ManifestError):
        M.validate_manifest(m)


# ---- Reuse rule ---------------------------------------------------------- #

def test_reusable_when_all_match():
    ref = parse_repo_url("pallets/flask")
    m = _ready_manifest(ref)
    ok, reason = M.is_reusable(m)
    assert ok, reason


def test_not_reusable_when_not_ready():
    ref = parse_repo_url("pallets/flask")
    m = _ready_manifest(ref)
    m["status"] = "fetching"
    ok, reason = M.is_reusable(m)
    assert not ok and "not ready" in reason


def test_not_reusable_on_embedding_model_change():
    ref = parse_repo_url("pallets/flask")
    m = _ready_manifest(ref)
    m["pipeline"]["embedding_model"] = "some-other-model:1.0"
    ok, reason = M.is_reusable(m)
    assert not ok and "embedding_model" in reason


def test_not_reusable_on_schema_bump():
    ref = parse_repo_url("pallets/flask")
    m = _ready_manifest(ref)
    m["schema_version"] = config.SCHEMA_VERSION + 1
    ok, reason = M.is_reusable(m)
    assert not ok and "schema_version" in reason


def test_not_reusable_on_chunker_bump():
    ref = parse_repo_url("pallets/flask")
    m = _ready_manifest(ref)
    m["pipeline"]["chunker_version"] = config.CHUNKER_VERSION + 1
    ok, reason = M.is_reusable(m)
    assert not ok and "chunker_version" in reason


# ---- Registry ------------------------------------------------------------ #

def test_registry_lists_and_finds(tmp_path):
    repos_dir = tmp_path / "repositories"
    ref_a = parse_repo_url("pallets/flask")
    ref_b = parse_repo_url("psf/requests")

    for ref in (ref_a, ref_b):
        ctx = RepositoryContext.for_ref(ref, repos_dir)
        ctx.ensure_dirs()
        M.write_manifest(ctx.manifest_path, _ready_manifest(ref))

    entries = registry.list_repositories(repos_dir)
    slugs = {e.slug for e in entries}
    assert slugs == {"pallets_flask", "psf_requests"}
    assert all(e.reusable for e in entries)

    found = registry.find(ref_a, repos_dir)
    assert found is not None and found.full_name == "pallets/flask"
    assert registry.is_indexed_and_reusable(ref_a, repos_dir)

    missing = registry.find(parse_repo_url("nobody/nothing"), repos_dir)
    assert missing is None


def test_registry_skips_corrupt_manifest(tmp_path):
    repos_dir = tmp_path / "repositories"
    d = repos_dir / "broken_repo"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text("{ broken", encoding="utf-8")
    # Should not raise, just skip.
    assert registry.list_repositories(repos_dir) == []
