"""Graph-aware candidate expansion for multi-hop ("evolution") questions.

Evolution questions are the pipeline's weakest category (recall 0.30-0.38)
because the answer is *spread across linked records*: an issue reports a
problem, a PR closes it, commits implement it, a release ships it. Pure
similarity search retrieves whichever single record best matches the wording
and stops there — the rest of the story is one graph edge away and never gets
retrieved.

``process/linker.py`` already builds exactly that graph (``links.json``). This
module reads it and, for each strong candidate, pulls in its linked neighbours
so the generator sees the whole chain rather than one link of it.

Design constraints:
  * **Additive only.** Expansion appends neighbours to the candidate pool; it
    never reorders or drops what similarity search found. The reranker still
    decides the final ordering, so a useless neighbour is ranked away rather
    than displacing a good hit.
  * **Bounded.** Every added candidate costs a cross-encoder pass later, which
    is the dominant query latency (HANDOFF.md §3.2). Expansion is capped per
    seed and overall.
  * **Data only.** Reads ``links.json`` from the repository context; imports
    nothing from ``eval/`` and adds no new dependency.
"""
from __future__ import annotations

import json
from typing import Any

# Only the highest-ranked candidates are used as expansion seeds: neighbours of
# a weak match are usually noise, and each one costs a reranker pass.
DEFAULT_MAX_SEEDS = 6
DEFAULT_MAX_PER_SEED = 4
DEFAULT_MAX_TOTAL = 12


def _base_id(chunk_id: str) -> str:
    """Strip a split suffix: ``pr_101#2`` -> ``pr_101``."""
    return chunk_id.split("#", 1)[0]


def load_graph(path) -> dict[str, Any]:
    """Load ``links.json``; return an empty graph if it is missing/corrupt."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def build_neighbour_map(graph: dict[str, Any]) -> dict[str, list[str]]:
    """Map a base chunk_id to the base chunk_ids it is linked to.

    Edges are followed in both directions, because the useful hop depends on
    what similarity search happened to find first:

        issue_N   -> the PRs / commits that closed it
        pr_N      -> the issues it closes, its commits, the release that shipped it
        commit_S  -> the PR that carried it, the issues it closes
        release_T -> the PRs it shipped
    """
    neighbours: dict[str, set[str]] = {}

    def link(a: str, b: str) -> None:
        if not a or not b or a == b:
            return
        neighbours.setdefault(a, set()).add(b)
        neighbours.setdefault(b, set()).add(a)

    for num, info in (graph.get("issues") or {}).items():
        issue = f"issue_{num}"
        for pr in info.get("closed_by_prs", []) or []:
            link(issue, f"pr_{pr}")
        for sha in info.get("closed_by_commits", []) or []:
            link(issue, f"commit_{str(sha)[:12]}")

    for num, info in (graph.get("prs") or {}).items():
        pr = f"pr_{num}"
        for issue in info.get("closes_issues", []) or []:
            link(pr, f"issue_{issue}")
        for sha in info.get("commits", []) or []:
            link(pr, f"commit_{str(sha)[:12]}")
        rel = info.get("release")
        if rel:
            link(pr, f"release_{str(rel).replace(' ', '_')}")

    for sha, info in (graph.get("commits") or {}).items():
        commit = f"commit_{str(sha)[:12]}"
        if info.get("pr") is not None:
            link(commit, f"pr_{info['pr']}")
        for issue in info.get("closes_issues", []) or []:
            link(commit, f"issue_{issue}")

    return {k: sorted(v) for k, v in neighbours.items()}


def expand(
    pool_ids: list[str],
    neighbour_map: dict[str, list[str]],
    chunks_by_base: dict[str, list[str]],
    *,
    max_seeds: int = DEFAULT_MAX_SEEDS,
    max_per_seed: int = DEFAULT_MAX_PER_SEED,
    max_total: int = DEFAULT_MAX_TOTAL,
) -> list[str]:
    """Return ``pool_ids`` plus linked-neighbour chunk ids, order preserved.

    ``chunks_by_base`` maps a base id to the concrete chunk ids present in the
    index (a long record may have been split into ``pr_1#0``, ``pr_1#1``, ...).
    Neighbours that were never indexed are silently skipped.
    """
    if not pool_ids or not neighbour_map:
        return pool_ids

    present = set(pool_ids)
    added: list[str] = []

    for seed in pool_ids[:max_seeds]:
        if len(added) >= max_total:
            break
        per_seed = 0
        for nb_base in neighbour_map.get(_base_id(seed), []):
            if per_seed >= max_per_seed or len(added) >= max_total:
                break
            for cid in chunks_by_base.get(nb_base, []):
                if cid in present:
                    continue
                present.add(cid)
                added.append(cid)
                per_seed += 1
                break   # one chunk per neighbour record is enough context
    return pool_ids + added


def apply_topk_cap(
    ranked: list[dict[str, Any]],
    graph_ids: set[str],
    top_k: int,
    max_from_graph: int,
) -> list[dict[str, Any]]:
    """Take the top ``top_k`` of ``ranked``, allowing at most ``max_from_graph``
    graph-expanded chunks.

    Measured motivation: letting expansion compete freely for every slot raised
    recall but *lowered* nDCG, because the cross-encoder scores a plausible
    linked record above the actual gold evidence and displaces it. Reserving
    most slots for the similarity hits keeps the ranking quality that already
    worked, while still letting the strongest one or two neighbours in — which
    is where the multi-hop recall gain comes from.

    Relative order is preserved; skipped neighbours simply fall through to the
    next candidate.
    """
    out: list[dict[str, Any]] = []
    used = 0
    for chunk in ranked:
        if len(out) >= top_k:
            break
        if chunk.get("chunk_id") in graph_ids:
            if used >= max_from_graph:
                continue
            used += 1
        out.append(chunk)
    return out
