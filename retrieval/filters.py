"""Date-range / source-type filters applied to the candidate pool.

Filters are applied *before* merging (they shrink the candidate pool), never
as a post-hoc filter on the final answer — so a date-scoped question retrieves
only in-scope evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class RetrievalFilters:
    """Optional constraints on which chunks are retrievable."""

    source_types: tuple[str, ...] | None = None
    since: str | None = None          # ISO date lower bound (inclusive)
    until: str | None = None          # ISO date upper bound (inclusive)

    def is_empty(self) -> bool:
        return not (self.source_types or self.since or self.until)


def build_chroma_where(filters: RetrievalFilters | None) -> dict[str, Any] | None:
    """Translate filters into a Chroma ``where`` clause (or None)."""
    if filters is None or filters.is_empty():
        return None
    clauses: list[dict[str, Any]] = []
    if filters.source_types:
        clauses.append({"source_type": {"$in": list(filters.source_types)}})
    if filters.since:
        clauses.append({"date": {"$gte": filters.since}})
    if filters.until:
        clauses.append({"date": {"$lte": filters.until}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def make_predicate(
    filters: RetrievalFilters | None,
) -> Callable[[dict[str, Any]], bool]:
    """A Python predicate mirroring the Chroma filter, for BM25 pre-filtering."""
    if filters is None or filters.is_empty():
        return lambda chunk: True

    def _pred(chunk: dict[str, Any]) -> bool:
        if filters.source_types and chunk.get("source_type") not in filters.source_types:
            return False
        date = chunk.get("date") or ""
        # Chunks lacking a date are only excluded when a bound is set.
        if filters.since:
            if not date or date < filters.since:
                return False
        if filters.until:
            if not date or date > filters.until:
                return False
        return True

    return _pred
