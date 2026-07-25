"""Phase C1: versioned, category-aware golden-set schema, loader, and validator.

One JSONL file per benchmark repository at ``eval/datasets/<owner>_<name>.jsonl``
(JSONL, not YAML — PyYAML is only a transitive dependency here and is not pinned,
so relying on it would break a clean rebuild; JSONL is standard-library and
matches the existing ``chunks.jsonl`` convention).

Each line is one object:
    id            stable identifier (unique within the file)
    question      the natural-language question
    query_type    factual | causal | cross_commit | evolution | unanswerable
    ground_truth  human-written reference answer (ABSENT/null for unanswerable)
    evidence      list of ground-truth chunk_ids / PR numbers / commit SHAs
                  (empty for unanswerable)
    notes         optional provenance string

The validator runs before any evaluation and fails loudly on duplicate ids or
malformed entries, listing every problem it found.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VALID_QUERY_TYPES = {"factual", "causal", "cross_commit", "evolution",
                     "unanswerable"}


class GoldenSetError(ValueError):
    """Raised when a golden set is malformed (lists every problem found)."""


@dataclass
class GoldEntry:
    id: str
    question: str
    query_type: str
    ground_truth: str | None = None
    evidence: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def is_unanswerable(self) -> bool:
        return self.query_type == "unanswerable"

    def to_dict(self) -> dict[str, Any]:
        d = {"id": self.id, "question": self.question,
             "query_type": self.query_type, "evidence": self.evidence}
        if self.ground_truth is not None:
            d["ground_truth"] = self.ground_truth
        if self.notes:
            d["notes"] = self.notes
        return d


def load_golden_set(path: str | Path) -> list[GoldEntry]:
    """Load a golden set from JSONL. Raises GoldenSetError on unreadable JSON."""
    p = Path(path)
    if not p.exists():
        raise GoldenSetError(f"Golden set not found: {p}")
    entries: list[GoldEntry] = []
    problems: list[str] = []
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(f"line {lineno}: invalid JSON ({exc})")
            continue
        if not isinstance(obj, dict):
            problems.append(f"line {lineno}: not a JSON object")
            continue
        entries.append(GoldEntry(
            id=str(obj.get("id", "")),
            question=str(obj.get("question", "")),
            query_type=str(obj.get("query_type", "")),
            ground_truth=obj.get("ground_truth"),
            evidence=[str(e) for e in obj.get("evidence", []) or []],
            notes=str(obj.get("notes", "")),
        ))
    if problems:
        raise GoldenSetError("Malformed golden set:\n  " + "\n  ".join(problems))
    return entries


def validate_golden_set(entries: list[GoldEntry]) -> None:
    """Validate a loaded golden set. Raises GoldenSetError listing all issues."""
    problems: list[str] = []
    seen: set[str] = set()
    for i, e in enumerate(entries):
        tag = e.id or f"<entry #{i}>"
        if not e.id:
            problems.append(f"{tag}: missing 'id'")
        elif e.id in seen:
            problems.append(f"{tag}: duplicate id")
        seen.add(e.id)
        if not e.question:
            problems.append(f"{tag}: missing 'question'")
        if e.query_type not in VALID_QUERY_TYPES:
            problems.append(
                f"{tag}: invalid query_type {e.query_type!r} "
                f"(expected one of {sorted(VALID_QUERY_TYPES)})")
        if e.is_unanswerable:
            if e.ground_truth:
                problems.append(f"{tag}: unanswerable entries must NOT have a "
                                f"ground_truth answer")
        else:
            if not e.ground_truth:
                problems.append(f"{tag}: answerable entries require a "
                                f"non-empty 'ground_truth'")
            if not e.evidence:
                problems.append(f"{tag}: answerable entries require non-empty "
                                f"'evidence'")
    if problems:
        raise GoldenSetError(
            f"Golden set failed validation ({len(problems)} problem(s)):\n  "
            + "\n  ".join(problems))


def category_distribution(entries: list[GoldEntry]) -> dict[str, int]:
    """Count entries per query_type (for reporting)."""
    dist: dict[str, int] = {qt: 0 for qt in sorted(VALID_QUERY_TYPES)}
    for e in entries:
        dist[e.query_type] = dist.get(e.query_type, 0) + 1
    return dist
