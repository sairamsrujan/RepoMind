"""Resumable ingestion checkpoint.

Persists per-source-type progress so an interrupted fetch resumes from the
last completed page (REST) or cursor (GraphQL) instead of restarting.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class Checkpoint:
    """A small JSON-backed progress tracker, one file per repository."""

    SOURCE_TYPES = ("commits", "prs", "issues", "releases")

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._state: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {st: self._blank() for st in self.SOURCE_TYPES}

    @staticmethod
    def _blank() -> dict[str, Any]:
        return {"done": False, "page": 0, "cursor": None, "count": 0}

    def get(self, source_type: str) -> dict[str, Any]:
        return self._state.setdefault(source_type, self._blank())

    def is_done(self, source_type: str) -> bool:
        return bool(self.get(source_type).get("done"))

    def update(self, source_type: str, **fields: Any) -> None:
        self.get(source_type).update(fields)
        self.save()

    def mark_done(self, source_type: str, count: int | None = None) -> None:
        entry = self.get(source_type)
        entry["done"] = True
        if count is not None:
            entry["count"] = count
        self.save()

    def reset(self, source_type: str | None = None) -> None:
        if source_type is None:
            self._state = {st: self._blank() for st in self.SOURCE_TYPES}
        else:
            self._state[source_type] = self._blank()
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._state, indent=2), encoding="utf-8")

    def all_done(self) -> bool:
        return all(self.is_done(st) for st in self.SOURCE_TYPES)
