"""Auto-derive a gold Q/A set for any repository.

For each closed issue that was resolved by a linked, merged PR, synthesize a
question from the issue and record that PR plus its commits as the ground-truth
evidence. This yields an evaluation set for any repo with no manual labeling.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from core.context import RepositoryContext
from process import chunker


@dataclass
class GoldItem:
    question: str
    issue_number: int
    issue_title: str
    pr_numbers: list[int] = field(default_factory=list)
    evidence_chunk_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "issue_number": self.issue_number,
            "issue_title": self.issue_title,
            "pr_numbers": self.pr_numbers,
            "evidence_chunk_ids": self.evidence_chunk_ids,
        }


def _commit_chunk_id(sha: str) -> str:
    return f"commit_{sha[:12]}"


def synthesize_gold_set(ctx: RepositoryContext) -> list[GoldItem]:
    """Build gold items from the link graph + issue chunks."""
    if not ctx.links_path.exists():
        return []
    graph = json.loads(ctx.links_path.read_text(encoding="utf-8"))
    chunks = chunker.load_chunks(ctx)

    # issue number -> title, and set of existing chunk ids (for evidence sanity).
    issue_title: dict[int, str] = {}
    existing_ids = {c["chunk_id"] for c in chunks}
    existing_bases = {cid.split("#", 1)[0] for cid in existing_ids}
    for c in chunks:
        if c["source_type"] == "issue":
            try:
                issue_title[int(c["ref_id"])] = c.get("title", "")
            except (ValueError, TypeError):
                pass

    gold: list[GoldItem] = []
    prs = graph.get("prs", {})
    for issue_str, info in graph.get("issues", {}).items():
        try:
            issue_n = int(issue_str)
        except (ValueError, TypeError):
            continue
        resolving_prs = [p for p in info.get("closed_by_prs", [])
                         if str(p) in prs and prs[str(p)].get("merged_at")]
        if not resolving_prs:
            continue

        evidence: list[str] = []
        for p in resolving_prs:
            if f"pr_{p}" in existing_bases:
                evidence.append(f"pr_{p}")
            for sha in prs[str(p)].get("commits", []):
                cid = _commit_chunk_id(sha)
                if cid in existing_bases:
                    evidence.append(cid)
        for sha in info.get("closed_by_commits", []):
            cid = _commit_chunk_id(sha)
            if cid in existing_bases:
                evidence.append(cid)
        evidence = sorted(set(evidence))
        if not evidence:
            continue

        title = issue_title.get(issue_n, f"issue #{issue_n}")
        question = (
            f"How was the issue '{title}' resolved, and what change addressed it?"
        )
        gold.append(GoldItem(
            question=question, issue_number=issue_n, issue_title=title,
            pr_numbers=sorted(resolving_prs), evidence_chunk_ids=evidence,
        ))

    return gold


def save_gold_set(ctx: RepositoryContext, gold: list[GoldItem]) -> None:
    path = ctx.base_dir / "gold_questions.json"
    path.write_text(json.dumps([g.to_dict() for g in gold], indent=2),
                    encoding="utf-8")
