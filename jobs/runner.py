"""Background ingestion runner: fetch -> chunk -> link -> embed -> index.

Runs as a separate process (``python -m jobs.runner <repo> [months]``) so the
Streamlit UI thread never blocks. Progress is reported through ``jobs/status``;
a per-repository lock file prevents two concurrent builds of the same repo from
corrupting each other.

Degraded mode: if PR ingestion fails (e.g. no GITHUB_TOKEN — GitHub's GraphQL
API requires auth), the run records a warning and still builds an index from
whatever sources succeeded, rather than aborting entirely.
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure project root importable when spawned via `python -m jobs.runner`.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from core import manifest as manifest_mod
from core.context import RepositoryContext
from core.repo_url import parse_repo_url
from ingest.checkpoint import Checkpoint
from ingest.fetch_commits import fetch_commits
from ingest.fetch_issues import fetch_issues
from ingest.fetch_prs import fetch_prs
from ingest.fetch_releases import fetch_releases
from ingest.github_client import GitHubClient, GitHubError, RateLimitExceeded
from index.embedder import OllamaEmbedder
from index.vector_store import build_index
from jobs import status as status_mod
from process import chunker, linker


def compute_window(months: int | None = None) -> tuple[str, str]:
    """Return (since, until) ISO timestamps for the last ``months`` months."""
    months = months or config.DEFAULT_LOOKBACK_MONTHS
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=30 * months)
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    return since.strftime(fmt), now.strftime(fmt)


# --------------------------------------------------------------------------- #
# Lock
# --------------------------------------------------------------------------- #
class LockError(RuntimeError):
    pass


class RepoLock:
    """A best-effort per-repository file lock (PID-stamped, staleness-aware)."""

    STALE_SECONDS = 3600

    def __init__(self, ctx: RepositoryContext):
        self.path = ctx.lock_path

    def acquire(self) -> "RepoLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            age = time.time() - self.path.stat().st_mtime
            if age < self.STALE_SECONDS and self._holder_alive():
                raise LockError(
                    f"Another ingestion is already running for this repo "
                    f"(lock: {self.path})."
                )
            # Stale lock -> steal it.
            self.path.unlink(missing_ok=True)
        fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return self

    def _holder_alive(self) -> bool:
        try:
            pid = int(self.path.read_text().strip() or "0")
        except (ValueError, OSError):
            return False
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def release(self) -> None:
        self.path.unlink(missing_ok=True)

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run_ingestion(
    repo: str,
    months: int | None = None,
    repositories_dir: Path | None = None,
) -> dict:
    """Run the full ingestion+index pipeline for ``repo``. Returns manifest."""
    ref = parse_repo_url(repo)
    ctx = RepositoryContext.for_ref(ref, repositories_dir).ensure_dirs()
    since, until = compute_window(months)
    warnings: list[str] = []

    def _status(stage, msg="", **extra):
        status_mod.write_status(ctx.status_path, stage, msg,
                                warnings=warnings, **extra)

    lock = RepoLock(ctx)
    try:
        lock.acquire()
    except LockError as exc:
        _status("failed", str(exc), error=str(exc))
        raise

    try:
        # Manifest: mark pending/fetching.
        m = manifest_mod.new_manifest(ref)
        m["coverage"]["since"] = since
        m["coverage"]["until"] = until
        m["status"] = "fetching"
        manifest_mod.write_manifest(ctx.manifest_path, m)
        _status("fetching", f"Fetching {ref.full_name} ({since[:10]}..{until[:10]})")

        client = GitHubClient(etag_cache_path=ctx.raw_dir / "etags.json")
        # Validate repo exists / get metadata (also a clear 404 path).
        try:
            meta = client.get_repo(ref.owner, ref.name)
            m["repo"]["github_id"] = meta.get("id", 0)
            m["repo"]["default_branch"] = meta.get("default_branch", "")
            manifest_mod.write_manifest(ctx.manifest_path, m)
        except GitHubError as exc:
            warnings.append(f"repo metadata: {exc}")

        cp = Checkpoint(ctx.checkpoint_path)
        counts = {}
        fetchers = [
            ("commits", fetch_commits),
            ("issues", fetch_issues),
            ("releases", fetch_releases),
            ("prs", fetch_prs),
        ]
        def _fetch_progress(msg: str) -> None:
            live = {n: cp.get(n)["count"] for n in Checkpoint.SOURCE_TYPES}
            _status("fetching", msg, live_counts=live)

        succeeded = 0
        rate_limited = False
        for name, fn in fetchers:
            try:
                counts[name] = fn(client, ctx, ref, since, until, cp,
                                  _fetch_progress)
                succeeded += 1
            except RateLimitExceeded as exc:
                rate_limited = True
                warnings.append(
                    f"{name}: hit GitHub rate limit — partial data fetched. "
                    f"Re-index this repo to resume from where it stopped."
                )
                _status("fetching", f"{name} paused (rate limit)")
            except GitHubError as exc:
                warnings.append(f"{name} fetch failed: {exc}")
                _status("fetching", f"{name} skipped: {exc}")
        if succeeded == 0:
            hint = ("Wait for the rate-limit reset and re-index to resume."
                    if rate_limited else "Check GITHUB_TOKEN and the repo URL.")
            msg = f"Could not fetch any data. {hint}"
            _status("failed", msg, error=msg)
            m["status"] = "failed"
            manifest_mod.write_manifest(ctx.manifest_path, m)
            raise GitHubError(msg)

        # Chunk.
        _status("chunking", "Chunking raw data")
        n_chunks = chunker.chunk_repository(ctx)
        if n_chunks == 0:
            msg = "No chunks produced (empty date window?)."
            _status("failed", msg, error=msg)
            m["status"] = "failed"
            manifest_mod.write_manifest(ctx.manifest_path, m)
            raise RuntimeError(msg)

        # Link.
        _status("linking", "Building relationship graph")
        linker.link_repository(ctx)

        # Embed + index.
        _status("embedding", f"Embedding {n_chunks} chunks")
        chunks = chunker.load_chunks(ctx)
        stats = build_index(
            ctx, chunks, OllamaEmbedder(),
            progress=lambda msg: _status("embedding", msg),
        )
        _status("indexing", "Finalizing index")

        # Manifest is set to ready inside build_index; refresh coverage window.
        m = manifest_mod.read_manifest(ctx.manifest_path)
        m["coverage"]["since"] = since
        m["coverage"]["until"] = until
        manifest_mod.write_manifest(ctx.manifest_path, m)

        ready_msg = f"Indexed {stats['chunks']} chunks"
        if rate_limited:
            ready_msg += " (partial — re-index to fetch the rest once the " \
                         "rate limit resets)"
        _status("ready", ready_msg, stats=stats, counts=counts)
        return m
    except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
        _status("failed", f"Ingestion failed: {exc}", error=str(exc))
        raise
    finally:
        lock.release()


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: python -m jobs.runner <repo-url-or-owner/name> [months]")
        return 2
    repo = argv[0]
    months = int(argv[1]) if len(argv) > 1 else None
    try:
        run_ingestion(repo, months)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
