"""Tests for the ingestion runner's pure pieces: date window + repo lock."""
from __future__ import annotations

import pytest

from core.context import RepositoryContext
from core.repo_url import parse_repo_url
from jobs.runner import LockError, RepoLock, compute_window


def test_compute_window_orders_and_formats():
    since, until = compute_window(3)
    assert since < until
    assert since.endswith("Z") and until.endswith("Z")
    assert len(since) == 20  # YYYY-MM-DDTHH:MM:SSZ


def test_repo_lock_blocks_concurrent(tmp_path):
    ref = parse_repo_url("acme/widgets")
    ctx = RepositoryContext.for_ref(ref, tmp_path / "repositories").ensure_dirs()

    lock1 = RepoLock(ctx).acquire()
    try:
        with pytest.raises(LockError):
            RepoLock(ctx).acquire()
    finally:
        lock1.release()

    # After release, a new lock can be acquired.
    lock2 = RepoLock(ctx).acquire()
    lock2.release()


def test_repo_lock_steals_stale(tmp_path):
    ref = parse_repo_url("acme/widgets")
    ctx = RepositoryContext.for_ref(ref, tmp_path / "repositories").ensure_dirs()
    # Write a lock owned by a definitely-dead PID.
    ctx.lock_path.write_text("999999999")
    # Should be treated as stale (holder not alive) and stealable.
    lock = RepoLock(ctx).acquire()
    lock.release()
