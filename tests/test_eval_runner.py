"""Phase C tests: golden-set schema/validator + evaluation-runner helpers."""
from __future__ import annotations

import json

import pytest
import requests

import config
from eval import golden_set, run as eval_run


# --------------------------------------------------------------------------- #
# Golden set: loader + validator
# --------------------------------------------------------------------------- #
def test_load_and_validate_sample_dataset():
    entries = golden_set.load_golden_set("eval/datasets/acme_widgets.jsonl")
    golden_set.validate_golden_set(entries)          # must not raise
    assert len(entries) == 5
    dist = golden_set.category_distribution(entries)
    assert dist["unanswerable"] == 1 and dist["factual"] == 1


def test_unanswerable_entry_has_no_ground_truth():
    entries = golden_set.load_golden_set("eval/datasets/acme_widgets.jsonl")
    un = next(e for e in entries if e.query_type == "unanswerable")
    assert un.is_unanswerable
    assert un.ground_truth in (None, "")
    assert un.evidence == []


def test_validator_flags_duplicate_ids():
    entries = [
        golden_set.GoldEntry("x", "q1", "factual", "a", ["pr_1"]),
        golden_set.GoldEntry("x", "q2", "causal", "a", ["pr_2"]),
    ]
    with pytest.raises(golden_set.GoldenSetError) as exc:
        golden_set.validate_golden_set(entries)
    assert "duplicate id" in str(exc.value)


def test_validator_flags_bad_type_and_missing_fields():
    entries = [
        golden_set.GoldEntry("a", "", "nonsense", None, []),
        golden_set.GoldEntry("b", "q", "factual", "", []),   # answerable, no GT
    ]
    with pytest.raises(golden_set.GoldenSetError) as exc:
        golden_set.validate_golden_set(entries)
    msg = str(exc.value)
    assert "invalid query_type" in msg and "ground_truth" in msg


def test_validator_flags_unanswerable_with_answer():
    entries = [golden_set.GoldEntry("u", "q", "unanswerable", "should not exist", [])]
    with pytest.raises(golden_set.GoldenSetError):
        golden_set.validate_golden_set(entries)


def test_load_rejects_malformed_json(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"id":"a"}\n{ not json\n', encoding="utf-8")
    with pytest.raises(golden_set.GoldenSetError):
        golden_set.load_golden_set(p)


# --------------------------------------------------------------------------- #
# Runner helpers (pure)
# --------------------------------------------------------------------------- #
def test_canonical_id_normalises():
    assert eval_run.canonical_id("commit_c0ffee100000") == "commit_c0ffee1"
    assert eval_run.canonical_id("pr_101#2") == "pr_101"
    assert eval_run.canonical_id("42") == "pr_42"
    assert eval_run.canonical_id("issue_2") == "issue_2"


def test_looks_like_abstention():
    assert eval_run.looks_like_abstention("The evidence does not contain that.")
    assert eval_run.looks_like_abstention("This is outside the indexed window.")
    assert not eval_run.looks_like_abstention("PR #101 fixed the crash.")


def test_did_abstain_signals():
    from types import SimpleNamespace
    assert eval_run.did_abstain(SimpleNamespace(empty=True))
    assert eval_run.did_abstain(SimpleNamespace(empty=False, refusal=True))
    assert eval_run.did_abstain(
        SimpleNamespace(empty=False, refusal=False, guard_pass=False, text="x"))
    assert eval_run.did_abstain(
        SimpleNamespace(empty=False, refusal=False, guard_pass=True,
                        text="No information about that here."))
    assert not eval_run.did_abstain(
        SimpleNamespace(empty=False, refusal=False, guard_pass=True,
                        text="PR #101 fixed the crash."))


def test_cached_judge_hits_cache(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_make_judge():
        def _judge(prompt):
            calls["n"] += 1
            return '{"faithfulness": 1.0, "answer_relevancy": 1.0}'
        return _judge

    monkeypatch.setattr(eval_run.metrics, "make_judge", fake_make_judge)
    cj = eval_run.CachedJudge(tmp_path / "cache.json", sleep_seconds=0.0)
    a = cj("some prompt")
    b = cj("some prompt")        # served from cache -> no second call
    assert a == b and calls["n"] == 1
    # Persisted, so a fresh instance also hits cache.
    cj2 = eval_run.CachedJudge(tmp_path / "cache.json", sleep_seconds=0.0)
    cj2("some prompt")
    assert calls["n"] == 1


def test_aggregate_breaks_down_by_query_type():
    rows = [
        {"id": "1", "query_type": "factual", "recall_at_k": 1.0, "mrr": 1.0,
         "latency_ms": 100.0},
        {"id": "2", "query_type": "factual", "recall_at_k": 0.0, "mrr": 0.0,
         "latency_ms": 200.0},
        {"id": "3", "query_type": "unanswerable", "correct_abstention": True,
         "latency_ms": 50.0},
    ]
    agg = eval_run.aggregate(rows)
    assert agg["factual"]["n"] == 2
    assert agg["factual"]["recall_at_k"] == 0.5
    assert agg["unanswerable"]["abstention_accuracy"] == 1.0


# --------------------------------------------------------------------------- #
# Live end-to-end (needs Ollama + an indexed acme/widgets)
# --------------------------------------------------------------------------- #
def _ready(slug: str) -> bool:
    from core.repo_url import parse_repo_url
    from core.context import RepositoryContext
    try:
        requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=3)
    except requests.RequestException:
        return False
    ctx = RepositoryContext.for_ref(parse_repo_url(slug))
    return ctx.manifest_path.exists()


@pytest.mark.skipif(not _ready("acme/widgets"),
                    reason="Ollama down or acme/widgets not indexed")
def test_eval_run_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "ollama")
    rc = eval_run.run(
        repo="acme/widgets", dataset="eval/datasets/acme_widgets.jsonl",
        metrics_list=["citation_precision", "recall_at_k"],  # skip judge = fast
        subset="full", out_dir=tmp_path / "out", sleep=0.0, limit=None)
    assert rc == 0
    results = json.loads((tmp_path / "out" / "results.json").read_text())
    assert "unanswerable" in results["by_query_type"]
    assert results["by_query_type"]["unanswerable"]["n"] == 1
    # The abstention metric must be present for the unanswerable category.
    assert "abstention_accuracy" in results["by_query_type"]["unanswerable"]
