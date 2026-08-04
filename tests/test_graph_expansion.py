"""Tests for graph-aware candidate expansion (multi-hop retrieval)."""
from __future__ import annotations

import json

import pytest

from retrieval import graph_expansion as gx

# A small but realistic slice of links.json: issue 1 was closed by PR 101,
# which carried two commits and shipped in v1.2.0.
GRAPH = {
    "issues": {
        "1": {"closed_by_prs": [101], "closed_by_commits": ["deadbeef00001111"]},
        "2": {"closed_by_prs": [], "closed_by_commits": []},
    },
    "prs": {
        "101": {"closes_issues": [1], "commits": ["c0ffee1122334455",
                                                  "deadbeef00001111"],
                "merged_at": "2024-03-05T00:00:00Z", "release": "v1.2.0"},
    },
    "commits": {
        "c0ffee1122334455": {"pr": 101, "closes_issues": []},
        "deadbeef00001111": {"pr": 101, "closes_issues": [1]},
    },
    "releases": {"v1.2.0": {"published_at": "2024-03-10T00:00:00Z", "prs": [101]}},
}


def _chunks_by_base(*base_ids: str) -> dict[str, list[str]]:
    return {b: [b] for b in base_ids}


def test_neighbour_map_links_issue_to_pr_and_commits():
    nb = gx.build_neighbour_map(GRAPH)
    assert "pr_101" in nb["issue_1"]
    assert "commit_deadbeef0000" in nb["issue_1"]


def test_edges_are_bidirectional():
    """Which hop is useful depends on what similarity search found first."""
    nb = gx.build_neighbour_map(GRAPH)
    assert "issue_1" in nb["pr_101"]
    assert "pr_101" in nb["commit_c0ffee112233"]


def test_pr_links_to_its_release():
    nb = gx.build_neighbour_map(GRAPH)
    assert "release_v1.2.0" in nb["pr_101"]


def test_commit_sha_is_truncated_to_chunk_id_form():
    """Chunk ids key commits on the 12-char sha (see process/chunker.py)."""
    nb = gx.build_neighbour_map(GRAPH)
    assert all(len(n) == len("commit_") + 12
               for n in nb["pr_101"] if n.startswith("commit_"))


def test_expand_pulls_in_the_linked_chain():
    """The core multi-hop win: retrieving the issue surfaces the PR that fixed it."""
    nb = gx.build_neighbour_map(GRAPH)
    by_base = _chunks_by_base("issue_1", "pr_101", "commit_deadbeef0000",
                              "commit_c0ffee112233", "release_v1.2.0")
    out = gx.expand(["issue_1"], nb, by_base)
    assert out[0] == "issue_1", "original candidates must stay first"
    assert "pr_101" in out
    assert len(out) > 1


def test_expand_is_purely_additive():
    """Expansion must never drop or reorder what similarity search returned."""
    nb = gx.build_neighbour_map(GRAPH)
    by_base = _chunks_by_base("issue_1", "pr_101")
    pool = ["issue_1", "issue_2"]
    out = gx.expand(pool, nb, by_base)
    assert out[:len(pool)] == pool


def test_expand_never_duplicates_existing_candidates():
    nb = gx.build_neighbour_map(GRAPH)
    by_base = _chunks_by_base("issue_1", "pr_101")
    out = gx.expand(["issue_1", "pr_101"], nb, by_base)
    assert len(out) == len(set(out))


def test_expand_skips_neighbours_that_were_never_indexed():
    """A link may point at a record outside the indexed date window."""
    nb = gx.build_neighbour_map(GRAPH)
    by_base = _chunks_by_base("issue_1")     # pr_101 not in the index
    out = gx.expand(["issue_1"], nb, by_base)
    assert out == ["issue_1"]


def test_expand_respects_max_total():
    """Every added candidate costs a cross-encoder pass, so the cap must hold."""
    nb = gx.build_neighbour_map(GRAPH)
    by_base = _chunks_by_base("issue_1", "pr_101", "commit_deadbeef0000",
                              "commit_c0ffee112233", "release_v1.2.0")
    out = gx.expand(["issue_1"], nb, by_base, max_total=1)
    assert len(out) == 2          # the seed plus exactly one neighbour


def test_expand_respects_max_seeds():
    nb = gx.build_neighbour_map(GRAPH)
    by_base = _chunks_by_base("issue_1", "pr_101", "commit_deadbeef0000")
    # With zero seeds considered, nothing can be added.
    out = gx.expand(["issue_1"], nb, by_base, max_seeds=0)
    assert out == ["issue_1"]


def test_split_chunk_ids_resolve_to_their_base():
    """A long PR split into pr_101#0/#1 must still be reachable as 'pr_101'."""
    nb = gx.build_neighbour_map(GRAPH)
    by_base = {"issue_1": ["issue_1"], "pr_101": ["pr_101#0", "pr_101#1"]}
    out = gx.expand(["issue_1"], nb, by_base)
    assert "pr_101#0" in out
    assert "pr_101#1" not in out, "one chunk per linked record is enough context"


def test_seed_with_split_suffix_still_matches():
    nb = gx.build_neighbour_map(GRAPH)
    by_base = _chunks_by_base("pr_101", "issue_1")
    out = gx.expand(["issue_1#2"], nb, by_base)
    assert "pr_101" in out


def test_empty_and_malformed_inputs_are_safe():
    assert gx.expand([], {}, {}) == []
    assert gx.expand(["a"], {}, {}) == ["a"]
    assert gx.build_neighbour_map({}) == {}
    assert gx.build_neighbour_map({"issues": None, "prs": None}) == {}


def test_load_graph_tolerates_missing_or_corrupt_file(tmp_path):
    """A repo indexed before the linker existed must not break retrieval."""
    missing = tmp_path / "nope.json"
    assert gx.load_graph(missing) == {}

    corrupt = tmp_path / "links.json"
    corrupt.write_text("{not json", encoding="utf-8")
    assert gx.load_graph(corrupt) == {}

    good = tmp_path / "good.json"
    good.write_text(json.dumps(GRAPH), encoding="utf-8")
    assert gx.load_graph(good)["prs"]["101"]["release"] == "v1.2.0"


def test_real_repository_graph_produces_neighbours():
    """Sanity-check against a real indexed repo, if one is present."""
    import config
    from core.context import RepositoryContext
    from core.repo_url import parse_repo_url

    ctx = RepositoryContext.for_ref(parse_repo_url("pallets/click"))
    if not ctx.links_path.exists():
        pytest.skip("pallets/click not indexed")
    nb = gx.build_neighbour_map(gx.load_graph(ctx.links_path))
    assert nb, "a real repo with links should yield neighbours"
    # Every mapped id should look like a chunk base id.
    assert all(k.split("_", 1)[0] in {"issue", "pr", "commit", "release"}
               for k in list(nb)[:50])
