"""Deleting an indexed repository.

This removes a directory tree, so the guardrails matter more than the happy
path: it must be impossible to escape ``repositories/`` via traversal, an
absolute path, or a symlink.
"""
from __future__ import annotations

import pytest

from core import registry
from core.registry import DeleteError


def _make_repo(base, slug, size=1024):
    d = base / slug
    (d / "raw" / "commits").mkdir(parents=True)
    (d / "manifest.json").write_text("{}", encoding="utf-8")
    (d / "raw" / "commits" / "page1.json").write_text("x" * size, encoding="utf-8")
    return d


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #
def test_deletes_repo_and_reports_bytes_freed(tmp_path):
    d = _make_repo(tmp_path, "acme_widgets", size=500)
    freed = registry.delete_repository("acme_widgets", repositories_dir=tmp_path)
    assert not d.exists()
    assert freed >= 500                     # counted the data it removed


def test_deletes_only_the_named_repo(tmp_path):
    keep = _make_repo(tmp_path, "keep_me")
    _make_repo(tmp_path, "delete_me")
    registry.delete_repository("delete_me", repositories_dir=tmp_path)
    assert keep.exists()                    # siblings untouched
    assert not (tmp_path / "delete_me").exists()


def test_repo_disappears_from_registry_listing(tmp_path):
    import json
    for slug, owner in (("a_one", "a"), ("b_two", "b")):
        d = tmp_path / slug
        d.mkdir(parents=True)
        (d / "manifest.json").write_text(json.dumps({
            "schema_version": 1,
            "repo": {"owner": owner, "name": slug.split("_")[1], "slug": slug},
            "coverage": {}, "pipeline": {}, "stats": {},
            "status": "ready", "updated_at": "2024-01-01",
        }), encoding="utf-8")

    before = {e.slug for e in registry.list_repositories(repositories_dir=tmp_path)}
    assert {"a_one", "b_two"} <= before
    registry.delete_repository("a_one", repositories_dir=tmp_path)
    after = {e.slug for e in registry.list_repositories(repositories_dir=tmp_path)}
    assert "a_one" not in after and "b_two" in after


# --------------------------------------------------------------------------- #
# Guardrails — none of these may delete anything
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [
    "", ".", "..", "../secrets", "../../etc", "a/b", "a\\b", "/etc",
])
def test_rejects_unsafe_names(tmp_path, bad):
    outside = tmp_path.parent / "must_survive.txt"
    outside.write_text("important", encoding="utf-8")
    with pytest.raises(DeleteError):
        registry.delete_repository(bad, repositories_dir=tmp_path)
    assert outside.exists()                 # nothing outside was touched


def test_rejects_missing_repo(tmp_path):
    with pytest.raises(DeleteError, match="No such indexed repository"):
        registry.delete_repository("never_indexed", repositories_dir=tmp_path)


def test_rejects_a_file_not_a_directory(tmp_path):
    (tmp_path / "just_a_file").write_text("x", encoding="utf-8")
    with pytest.raises(DeleteError):
        registry.delete_repository("just_a_file", repositories_dir=tmp_path)
    assert (tmp_path / "just_a_file").exists()


def test_symlink_escaping_the_base_is_refused(tmp_path):
    """A symlink pointing outside repositories/ must not delete the target."""
    secret_dir = tmp_path.parent / "secret_data"
    secret_dir.mkdir(exist_ok=True)
    (secret_dir / "keep.txt").write_text("do not delete", encoding="utf-8")

    link = tmp_path / "sneaky"
    link.symlink_to(secret_dir, target_is_directory=True)

    with pytest.raises(DeleteError):
        registry.delete_repository("sneaky", repositories_dir=tmp_path)
    assert (secret_dir / "keep.txt").exists()   # target survived
