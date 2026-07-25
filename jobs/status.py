"""Job status file that the Streamlit UI polls for live ingestion progress.

The runner writes ``job_status.json``; the UI reads it every second or two.
Stages mirror the manifest pipeline: fetching -> chunking -> embedding ->
indexing -> ready (or failed).
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.context import RepositoryContext

STAGES = ("pending", "fetching", "chunking", "linking", "embedding",
          "indexing", "ready", "failed")

# Rough progress weighting per stage, for a monotonic UI progress bar.
STAGE_PROGRESS = {
    "pending": 0.02, "fetching": 0.25, "chunking": 0.45, "linking": 0.55,
    "embedding": 0.75, "indexing": 0.9, "ready": 1.0, "failed": 1.0,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(
    path: str | os.PathLike,
    stage: str,
    message: str = "",
    error: str = "",
    **extra: Any,
) -> dict[str, Any]:
    """Atomically write the current job status. Returns the status dict."""
    status = {
        "stage": stage,
        "message": message,
        "progress": STAGE_PROGRESS.get(stage, 0.0),
        "error": error,
        "updated_at": _utc_now(),
        **extra,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".status.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(status, fh, indent=2)
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return status


def read_status(path: str | os.PathLike) -> dict[str, Any] | None:
    """Read the current job status, or None if none exists yet."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def status_for(ctx: RepositoryContext) -> dict[str, Any] | None:
    return read_status(ctx.status_path)
