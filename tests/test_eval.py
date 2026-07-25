"""Phase 8: retrieval/citation metrics, gold-set synthesis, judge dispatch."""
from __future__ import annotations

import math

import pytest
import requests

import config
from eval import metrics
from eval.synth_questions import synthesize_gold_set
from process import chunker, linker


def _ollama_up() -> bool:
    try:
        requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=3)
        return True
    except requests.RequestException:
        return False


# ---- retrieval metrics (pure) -------------------------------------------- #
def test_recall_at_k():
    retrieved = ["a", "b", "c", "d"]
    relevant = ["c", "e"]
    assert metrics.recall_at_k(retrieved, relevant, 4) == 0.5   # c found, e not
    assert metrics.recall_at_k(retrieved, relevant, 2) == 0.0   # c not in top-2
    assert metrics.recall_at_k(retrieved, ["a", "b"], 2) == 1.0


def test_mrr():
    assert metrics.mrr(["x", "y", "rel"], ["rel"]) == pytest.approx(1 / 3)
    assert metrics.mrr(["rel", "y"], ["rel"]) == 1.0
    assert metrics.mrr(["x", "y"], ["rel"]) == 0.0


def test_ndcg():
    # Relevant item at rank 1 -> perfect nDCG.
    assert metrics.ndcg_at_k(["rel", "x"], ["rel"], 2) == pytest.approx(1.0)
    # Relevant at rank 2 -> DCG = 1/log2(3), IDCG = 1 -> < 1.
    val = metrics.ndcg_at_k(["x", "rel"], ["rel"], 2)
    assert val == pytest.approx(1 / math.log2(3))


def test_metrics_normalize_split_chunk_ids():
    # A retrieved split chunk "pr_1#0" should match gold "pr_1".
    assert metrics.recall_at_k(["pr_1#0"], ["pr_1"], 1) == 1.0


# ---- citation metrics ---------------------------------------------------- #
def test_citation_precision_recall():
    cited = ["pr_1", "pr_2"]
    gold = ["pr_1", "pr_3"]
    p, r = metrics.citation_precision_recall(cited, gold)
    assert p == 0.5 and r == 0.5

    p2, r2 = metrics.citation_precision_recall([], ["pr_1"])
    assert p2 == 0.0 and r2 == 0.0


# ---- gold-set synthesis -------------------------------------------------- #
def test_synthesize_gold_set(fixture_repo):
    chunker.chunk_repository(fixture_repo)
    linker.link_repository(fixture_repo)
    gold = synthesize_gold_set(fixture_repo)
    # Issue #1 was closed by merged PR #101 -> a gold item must exist.
    assert any(g.issue_number == 1 for g in gold)
    g1 = next(g for g in gold if g.issue_number == 1)
    assert 101 in g1.pr_numbers
    assert any(e.startswith("pr_101") for e in g1.evidence_chunk_ids)
    assert "Crash on startup" in g1.question


# ---- judge dispatch ------------------------------------------------------ #
def test_make_judge_defaults_local_without_keys(monkeypatch):
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "groq")
    monkeypatch.setattr(config, "GROQ_API_KEY", None)
    # No key -> must not raise; returns a callable (local fallback).
    judge = metrics.make_judge()
    assert callable(judge)


@pytest.mark.skipif(not _ollama_up(), reason="Ollama not running")
def test_local_judge_scores_faithfulness(monkeypatch):
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "ollama")
    out = metrics.judge_faithfulness_relevancy(
        "What was fixed?",
        "The startup crash was fixed by null-checking config [pr_101].",
        ["Pull Request #101: Fix startup crash by null-checking config."],
    )
    # Faithfulness should be a valid score (grounded answer -> reasonably high).
    assert out["faithfulness"] is None or 0.0 <= out["faithfulness"] <= 1.0
    assert "ollama" in out["judge"]
