"""Tests for the golden-set auto-generator (pure parts + mocked LLM)."""
from __future__ import annotations

import pytest

from eval import generate_golden_set as gg
from eval.golden_set import GoldenSetError


def _chunk(cid, stype, title, text="some text here ok", date="2024-05-01",
           tokens=20):
    return {"chunk_id": cid, "source_type": stype, "title": title,
            "text": text, "date": date, "token_count": tokens,
            "github_url": f"https://example.com/{cid}"}


# --------------------------------------------------------------------------- #
# Category targets
# --------------------------------------------------------------------------- #
def test_category_targets_sum_exactly_to_n():
    for n in (5, 25, 50, 37):
        t = gg.category_targets(n)
        assert sum(t.values()) == n
        assert set(t) == set(gg.CATEGORY_WEIGHTS)


def test_category_targets_50_matches_documented_split():
    t = gg.category_targets(50)
    assert t == {"factual": 13, "causal": 8, "cross_commit": 12,
                 "evolution": 10, "unanswerable": 7}


# --------------------------------------------------------------------------- #
# Candidate selection
# --------------------------------------------------------------------------- #
def test_substantive_filters_noise_and_short_chunks():
    assert gg._is_substantive(_chunk("c1", "commit", "Fix crash on startup"))
    assert not gg._is_substantive(_chunk("c2", "commit", "Merge branch main"))
    assert not gg._is_substantive(_chunk("c3", "commit", "Bump version 1.2"))
    assert not gg._is_substantive(_chunk("c4", "commit", "Real title", tokens=3))


def test_causal_candidates_require_rationale_language():
    chunks = [
        _chunk("c1", "commit", "t", text="Fixed the crash because of a null"),
        _chunk("c2", "commit", "t", text="Add shiny button to toolbar"),
        _chunk("r1", "release", "t", text="fixes many things"),  # wrong type
    ]
    got = {c[0]["chunk_id"] for c in gg.causal_candidates(chunks)}
    assert got == {"c1"}


def test_cross_commit_clusters_from_link_graph():
    chunks = [
        _chunk("issue_7", "issue", "Crash report"),
        _chunk("pr_9", "pr", "Fix crash"),
        _chunk("commit_abcdef123456", "commit", "fix (#9)"),
    ]
    graph = {"issues": {"7": {"closed_by_prs": [9],
                              "closed_by_commits": ["abcdef123456"]}}}
    clusters = gg.cross_commit_candidates(chunks, graph)
    ids = {c["chunk_id"] for c in clusters[0]}
    assert {"issue_7", "pr_9", "commit_abcdef123456"} <= ids


def test_evolution_candidates_need_shared_keyword_across_dates():
    chunks = [
        _chunk("c1", "commit", "improve caching layer", date="2024-01-01"),
        _chunk("c2", "commit", "fix caching regression", date="2024-03-01"),
        _chunk("c3", "commit", "unrelated docs tweak", date="2024-03-02"),
    ]
    clusters = gg.evolution_candidates(chunks)
    assert any({c["chunk_id"] for c in cl} == {"c1", "c2"} for cl in clusters)
    # Same-day pairs don't count as evolution.
    same_day = [_chunk("a", "commit", "caching x", date="2024-01-01"),
                _chunk("b", "commit", "caching y", date="2024-01-01")]
    assert gg.evolution_candidates(same_day) == []


# --------------------------------------------------------------------------- #
# LLM-dependent paths, with the chat call mocked
# --------------------------------------------------------------------------- #
def test_synthesize_entry_parses_and_grounds(monkeypatch):
    monkeypatch.setattr(gg, "_ollama_chat", lambda *a, **k:
                        '{"question": "Why X?", "ground_truth": "Because Y."}')
    cluster = [_chunk("pr_1", "pr", "t"), _chunk("commit_abc", "commit", "t")]
    e = gg.synthesize_entry(cluster, "causal", "o/r", "m", 3)
    assert e.id == "auto-causal-003" and e.query_type == "causal"
    assert e.evidence == ["pr_1", "commit_abc"]
    assert e.ground_truth == "Because Y."


def test_synthesize_entry_rejects_bad_llm_output(monkeypatch):
    monkeypatch.setattr(gg, "_ollama_chat", lambda *a, **k: "not json at all")
    assert gg.synthesize_entry([_chunk("c", "pr", "t")], "factual",
                               "o/r", "m", 1) is None
    monkeypatch.setattr(gg, "_ollama_chat", lambda *a, **k:
                        '{"question": "", "ground_truth": ""}')
    assert gg.synthesize_entry([_chunk("c", "pr", "t")], "factual",
                               "o/r", "m", 1) is None


def test_extract_json_handles_wrapped_output():
    assert gg._extract_json('noise {"a": 1} noise') == {"a": 1}
    assert gg._extract_json('["q1", "q2"]') == ["q1", "q2"]
    assert gg._extract_json("nothing") is None


def test_save_golden_set_validates_before_writing(tmp_path):
    from eval.golden_set import GoldEntry
    bad = [GoldEntry("x", "q", "unanswerable", "must-not-have-answer", [])]
    out = tmp_path / "bad.jsonl"
    with pytest.raises(GoldenSetError):
        gg.save_golden_set(bad, out)
    assert not out.exists()  # nothing written on validation failure
