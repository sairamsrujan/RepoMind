"""Phase D tests: extended ablation (configs, CSV/JSON output, live run)."""
from __future__ import annotations

import csv
import json

import pytest
import requests

import config
from eval import ablation


def test_config_set_includes_channels_and_retry():
    names = {c["name"] for c in ablation.ABLATION_CONFIGS}
    assert "5.full+guard+retry" in names
    assert "dense-only" in names and "sparse-only" in names
    dense = next(c for c in ablation.ABLATION_CONFIGS if c["name"] == "dense-only")
    sparse = next(c for c in ablation.ABLATION_CONFIGS if c["name"] == "sparse-only")
    assert dense["sparse"] is False and dense["dense"] is True
    assert sparse["dense"] is False and sparse["sparse"] is True


def _fake_results():
    by_type = {
        "factual": {"n": 1, "recall_at_k": 1.0, "mrr": 1.0, "ndcg": 1.0,
                    "citation_precision": 0.5, "citation_recall": 0.5,
                    "faithfulness": 0.9, "answer_relevancy": 0.8,
                    "latency_ms_mean": 100.0, "latency_ms_p95": 120.0},
        "unanswerable": {"n": 1, "abstention_accuracy": 1.0, "hallucinated": 0,
                         "latency_ms_mean": 50.0, "latency_ms_p95": 50.0},
    }
    return {"acme/widgets": {"3.+MMR+reranker": {"by_type": by_type, "rows": []}}}


def test_write_ablation_outputs(tmp_path):
    ablation.write_ablation_outputs(_fake_results(), tmp_path)
    assert (tmp_path / "ablation.json").exists()
    assert (tmp_path / "ablation.csv").exists()

    data = json.loads((tmp_path / "ablation.json").read_text())
    assert "acme/widgets" in data

    rows = list(csv.DictReader((tmp_path / "ablation.csv").open()))
    header = rows[0].keys()
    for col in ("repo", "config", "query_type", "recall_at_k",
                "abstention_accuracy", "latency_ms_p95"):
        assert col in header
    # One CSV row per (repo, config, query_type).
    assert len(rows) == 2
    factual = next(r for r in rows if r["query_type"] == "factual")
    assert factual["recall_at_k"] == "1.0000"


def test_format_golden_ablation_runs():
    txt = ablation.format_golden_ablation(_fake_results())
    assert "Ablation — acme/widgets" in txt
    assert "3.+MMR+reranker" in txt


def _ready(slug: str) -> bool:
    from core.context import RepositoryContext
    from core.repo_url import parse_repo_url
    try:
        requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=3)
    except requests.RequestException:
        return False
    return RepositoryContext.for_ref(parse_repo_url(slug)).manifest_path.exists()


@pytest.mark.skipif(not _ready("acme/widgets"),
                    reason="Ollama down or acme/widgets not indexed")
def test_run_golden_ablation_live(monkeypatch):
    from core.context import RepositoryContext
    from core.repo_url import parse_repo_url
    from eval.golden_set import load_golden_set

    ctx = RepositoryContext.for_ref(parse_repo_url("acme/widgets"))
    entries = load_golden_set("eval/datasets/acme_widgets.jsonl")[:2]
    # Two cheap configs (no judge, no retry) to keep the test fast.
    cfgs = [c for c in ablation.ABLATION_CONFIGS
            if c["name"] in ("1.retrieval-only", "sparse-only")]
    judge_state: dict = {"available": True}
    res = ablation.run_golden_ablation(
        ctx, entries, ["citation_precision", "recall_at_k"], judge=None,
        judge_state=judge_state, configs=cfgs)
    assert set(res) == {"1.retrieval-only", "sparse-only"}
    for cname, data in res.items():
        assert "by_type" in data and data["rows"]
