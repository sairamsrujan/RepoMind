"""Reference validation guard.

Every chunk_id cited inline in an answer must correspond to a chunk that was
actually retrieved and shown to the model. Any citation that does not is flagged
as a fabricated reference.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from generation.answerer import extract_citations


def _base_id(chunk_id: str) -> str:
    """Strip a split suffix: ``pr_101#2`` -> ``pr_101``."""
    return chunk_id.split("#", 1)[0]


_MIN_SHA_PREFIX = 7  # git short-sha length; safe against accidental collisions


def _citation_matches(cid: str, retrieved_ids: set[str],
                      retrieved_bases: set[str]) -> bool:
    """Whether a cited id resolves to a retrieved chunk.

    Exact or base-id match, plus commit-sha *prefix* tolerance so a citation
    like ``commit_feed3003`` still matches the chunk id ``commit_feed30030000``
    (the chunker keys commits on the 12-char sha, but models often abbreviate).
    """
    base = _base_id(cid)
    if cid in retrieved_ids or base in retrieved_bases:
        return True
    if base.startswith("commit_"):
        sha = base[len("commit_"):]
        if len(sha) >= _MIN_SHA_PREFIX:
            for rb in retrieved_bases:
                if rb.startswith("commit_"):
                    rsha = rb[len("commit_"):]
                    if rsha.startswith(sha) or sha.startswith(rsha):
                        return True
    return False


@dataclass
class ReferenceReport:
    valid_citations: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    uncited: bool = False        # answer made claims but cited nothing

    @property
    def is_valid(self) -> bool:
        return not self.invalid_citations


def validate_references(
    answer_text: str, retrieved_chunks: list[dict[str, Any]]
) -> ReferenceReport:
    """Check that all inline citations resolve to retrieved chunks."""
    retrieved_ids = {c["chunk_id"] for c in retrieved_chunks}
    retrieved_bases = {_base_id(cid) for cid in retrieved_ids}

    cited = extract_citations(answer_text)
    valid, invalid = [], []
    for cid in cited:
        if _citation_matches(cid, retrieved_ids, retrieved_bases):
            valid.append(cid)
        else:
            invalid.append(cid)

    return ReferenceReport(
        valid_citations=valid,
        invalid_citations=invalid,
        uncited=(len(cited) == 0),
    )
