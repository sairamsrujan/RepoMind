"""Tests for the chunker (schema correctness) and linker (relationship graph)."""
from __future__ import annotations

import config
from process import chunker, linker

REQUIRED_KEYS = {
    "chunk_id", "source_type", "ref_id", "title", "author", "date",
    "github_url", "linked_refs", "text", "token_count",
}
VALID_SOURCE_TYPES = {"commit", "pr", "issue", "review", "release"}


def test_parse_issue_refs():
    assert chunker.parse_issue_refs("fixes #1 and see #42") == [1, 42]
    assert chunker.parse_closing_refs("Closes #7, resolves #8, mentions #9") == [7, 8]
    assert chunker.parse_issue_refs("") == []


def test_chunk_schema_correct(fixture_repo):
    n = chunker.chunk_repository(fixture_repo)
    assert n > 0
    chunks = chunker.load_chunks(fixture_repo)
    assert len(chunks) == n
    seen_ids = set()
    for c in chunks:
        assert REQUIRED_KEYS <= set(c), f"missing keys in {c}"
        assert c["source_type"] in VALID_SOURCE_TYPES
        assert c["chunk_id"] not in seen_ids, "chunk_id must be unique"
        seen_ids.add(c["chunk_id"])
        assert isinstance(c["linked_refs"], list)
        assert c["token_count"] >= 1
        assert c["token_count"] <= config.MAX_CHUNK_TOKENS + 1
        assert c["text"].strip()


def test_all_source_types_present(fixture_repo):
    chunker.chunk_repository(fixture_repo)
    chunks = chunker.load_chunks(fixture_repo)
    types = {c["source_type"] for c in chunks}
    # commit, pr, issue, review, release should all appear in the fixture.
    assert {"commit", "pr", "issue", "review", "release"} <= types


def test_long_text_is_split():
    long_text = "word " * 5000
    parts = chunker._make_chunks(
        base_id="issue_9", source_type="issue", ref_id="9", title="big",
        author="x", date="", url="", linked_refs=[], text=long_text,
    )
    assert len(parts) > 1
    assert all(p["token_count"] <= config.MAX_CHUNK_TOKENS + 1 for p in parts)
    # split chunk ids are suffixed and unique
    ids = [p["chunk_id"] for p in parts]
    assert ids[0] == "issue_9#0" and len(set(ids)) == len(ids)


def test_linker_finds_real_closes_link(fixture_repo):
    graph = linker.link_repository(fixture_repo)
    # Issue #1 is closed by PR #101 AND commit c0ffee1 (fixes #1).
    assert 1 in graph["issues"]
    assert 101 in graph["issues"][1]["closed_by_prs"]
    assert any(s.startswith("c0ffee1") for s in graph["issues"][1]["closed_by_commits"])
    assert linker.count_closes_links(graph) >= 1


def test_to_dot_renders_graph(fixture_repo):
    graph = linker.link_repository(fixture_repo)
    dot = linker.to_dot(graph)
    assert dot.startswith("digraph G {") and dot.rstrip().endswith("}")
    # Issue #1 closed by PR #101 -> both nodes and the edge must appear.
    assert '"issue_1"' in dot
    assert '"pr_101"' in dot
    assert '"pr_101" -> "issue_1"' in dot
    # Release edge from the merged PR.
    assert "release_v1.2.0" in dot


def test_to_dot_empty_graph_is_valid():
    dot = linker.to_dot({"issues": {}, "prs": {}, "commits": {}, "releases": {}})
    assert dot.startswith("digraph G {") and dot.rstrip().endswith("}")


def test_linker_pr_to_release_and_commits(fixture_repo):
    graph = linker.link_repository(fixture_repo)
    pr = graph["prs"][101]
    assert pr["closes_issues"] == [1]
    assert any(s.startswith("c0ffee1") for s in pr["commits"])
    # PR #101 merged 2024-03-03, first release after is v1.2.0 (2024-03-10).
    assert pr["release"] == "v1.2.0"
    assert 101 in graph["releases"]["v1.2.0"]["prs"]
