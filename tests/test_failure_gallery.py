"""Phase E tests: failure-gallery extraction, spread, and rendering."""
from __future__ import annotations

import json

from scripts import export_failure_gallery as fg


def test_from_metrics_categorises(tmp_path):
    p = tmp_path / "queries.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [
        {"empty": False, "guard_verdict": "fail",
         "guard_reason": "fabricated_citations=1", "question": "q1",
         "num_contradicted": 0},
        {"empty": False, "guard_verdict": "fail",
         "guard_reason": "contradicted_claims=1", "question": "q2",
         "num_contradicted": 1},
        {"empty": False, "guard_verdict": "pass", "guard_reason": "ok",
         "question": "q3"},
    ]), encoding="utf-8")
    cases = fg.from_metrics(p)
    cats = {c.category for c in cases}
    assert cats == {"fabricated_citation", "nli_contradiction"}
    assert len(cases) == 2      # the passing query is not a failure


def test_from_results_detects_miss_and_refusal(tmp_path):
    ds = tmp_path / "ds.jsonl"
    ds.write_text("\n".join(json.dumps(e) for e in [
        {"id": "a", "question": "Q-A", "query_type": "factual",
         "ground_truth": "answer A", "evidence": ["pr_1"]},
        {"id": "b", "question": "Q-B", "query_type": "causal",
         "ground_truth": "answer B", "evidence": ["pr_2"]},
    ]), encoding="utf-8")
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"rows": [
        {"id": "a", "query_type": "factual", "recall_at_k": 0.0,
         "guard_pass": True, "refusal": False, "answer": "some answer"},
        {"id": "b", "query_type": "causal", "recall_at_k": 1.0,
         "guard_pass": False, "refusal": False, "answer": "weak answer"},
    ]}), encoding="utf-8")

    cases = fg.from_results(results, ds)
    cats = {c.category for c in cases}
    assert "retrieval_miss" in cats            # row a: recall 0
    assert "incorrect_refusal" in cats         # row b: guard_pass False
    miss = next(c for c in cases if c.category == "retrieval_miss")
    assert miss.question == "Q-A" and miss.ground_truth == "answer A"


def test_unanswerable_rows_are_not_failures(tmp_path):
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"rows": [
        {"id": "u", "query_type": "unanswerable", "recall_at_k": 0.0,
         "guard_pass": False, "refusal": True, "answer": "declined"},
    ]}), encoding="utf-8")
    assert fg.from_results(results, None) == []


def test_select_spread_across_categories():
    cases = [
        fg.FailureCase("retrieval_miss", f"q{i}", "r", "g", "s", "eval")
        for i in range(4)
    ] + [fg.FailureCase("nli_contradiction", "n1", "r", "g", "s", "m"),
         fg.FailureCase("fabricated_citation", "f1", "r", "g", "s", "m")]
    picked = fg.select_spread(cases, limit=3)
    assert len(picked) == 3
    # Spread: should include the two singletons before piling on retrieval_miss.
    cats = {c.category for c in picked}
    assert "nli_contradiction" in cats and "fabricated_citation" in cats


def test_render_emits_a_catalogue_not_a_worksheet():
    """Every case carries its own evidence; nothing is left for a human to fill.

    The rendered file ships in the repository and the README links to it as a
    catalogue of real failures, so an empty field or an instruction addressed to
    the author must never appear in it.
    """
    cases = [fg.FailureCase("retrieval_miss", "Why X?", "returned", "truth",
                            "retrieval", "eval")]
    md = fg.render(cases)
    assert "# RepoMind — failure gallery" in md
    assert "[retrieval_miss]" in md
    for field in ("Stage that failed", "System returned", "Ground truth"):
        assert field in md
    assert "**Why:**" not in md
    assert "Fill in" not in md


def test_main_writes_file(tmp_path):
    metrics = tmp_path / "q.jsonl"
    metrics.write_text(json.dumps(
        {"empty": False, "guard_verdict": "fail",
         "guard_reason": "fabricated_citations=1", "question": "q1"}) + "\n",
        encoding="utf-8")
    out = tmp_path / "gallery.md"
    rc = fg.main(["--metrics", str(metrics), "--out", str(out), "--limit", "5"])
    assert rc == 0 and out.exists()
    assert "fabricated_citation" in out.read_text()
