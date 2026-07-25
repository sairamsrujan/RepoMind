"""Phase B tests: adaptive verification retry loop.

Uses fakes for the retriever / answerer / NLI verifier (no models, no network)
and the *real* reference validator, so guard outcomes are controlled precisely.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import config
import query_pipeline
from generation.answerer import extract_citations

CHUNKS = [{"chunk_id": "pr_1", "source_type": "pr", "text": "PR #1 fixes it."}]


class FakeRetriever:
    def __init__(self, per_call_chunks):
        self.per_call_chunks = per_call_chunks
        self.calls = []

    def retrieve(self, question, use_mmr=True, dense_k=None, sparse_k=None,
                 trace=None, **kw):
        self.calls.append({"use_mmr": use_mmr, "dense_k": dense_k,
                           "sparse_k": sparse_k})
        idx = min(len(self.calls) - 1, len(self.per_call_chunks) - 1)
        return self.per_call_chunks[idx]


class FakeAnswerer:
    def __init__(self, texts):
        self.texts = texts
        self.calls = 0

    def answer(self, question, chunks, since, until):
        text = self.texts[min(self.calls, len(self.texts) - 1)]
        self.calls += 1
        return SimpleNamespace(text=text, cited_chunk_ids=extract_citations(text))


def _nli_report(grounded: bool):
    return SimpleNamespace(
        is_grounded=grounded,
        contradicted=[] if grounded else [SimpleNamespace(claim="x", contradiction=0.9)],
        unverified=[],
    )


class FakeNLI:
    def __init__(self, grounded_seq):
        self.grounded_seq = grounded_seq
        self.calls = 0

    def verify(self, text, chunks):
        g = self.grounded_seq[min(self.calls, len(self.grounded_seq) - 1)]
        self.calls += 1
        return _nli_report(g)


def _answer(retriever, answerer, nli):
    return query_pipeline.answer_query(retriever, answerer, nli,
                                       "why?", "2024-01-01", "2024-12-31")


# --------------------------------------------------------------------------- #
def test_guard_pass_does_not_retry(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_ADAPTIVE_RETRY", True)
    r = FakeRetriever([CHUNKS])
    a = FakeAnswerer(["Fixed by [pr_1]."])
    nli = FakeNLI([True])
    pr = _answer(r, a, nli)
    assert len(r.calls) == 1                 # no retry
    assert pr.guard_pass and not pr.retry_attempted and not pr.refusal
    assert pr.text == "Fixed by [pr_1]."


def test_guard_fail_triggers_one_widened_retry(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_ADAPTIVE_RETRY", True)
    r = FakeRetriever([CHUNKS, CHUNKS])
    a = FakeAnswerer(["Fixed by [pr_1].", "Fixed by [pr_1]."])
    nli = FakeNLI([False, True])             # fail then pass
    pr = _answer(r, a, nli)
    assert len(r.calls) == 2                 # exactly one retry
    # Retry must skip MMR and widen both pools.
    assert r.calls[1]["use_mmr"] is False
    assert r.calls[1]["dense_k"] == config.DENSE_TOP_K * config.RETRY_POOL_MULTIPLIER
    assert r.calls[1]["sparse_k"] == config.SPARSE_TOP_K * config.RETRY_POOL_MULTIPLIER
    assert pr.retry_attempted and pr.retry_succeeded and pr.guard_pass
    assert not pr.refusal


def test_double_failure_produces_refusal_not_unverified(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_ADAPTIVE_RETRY", True)
    r = FakeRetriever([CHUNKS, CHUNKS])
    a = FakeAnswerer(["Unverified claim [pr_1].", "Still unverified [pr_1]."])
    nli = FakeNLI([False, False])            # fail both
    pr = _answer(r, a, nli)
    assert len(r.calls) == 2                 # bounded at one retry
    assert pr.retry_attempted and not pr.retry_succeeded
    assert pr.refusal and not pr.guard_pass
    assert pr.text == query_pipeline.REFUSAL_TEXT
    # The model's unverified text must NOT be surfaced as the answer.
    assert "Unverified claim" not in pr.text
    assert "Still unverified" not in pr.text


def test_flag_off_is_identical_to_today(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_ADAPTIVE_RETRY", False)
    r = FakeRetriever([CHUNKS])
    a = FakeAnswerer(["Weak answer [pr_1]."])
    nli = FakeNLI([False])                   # guard fails
    pr = _answer(r, a, nli)
    assert len(r.calls) == 1                 # NO retry when flag off
    assert not pr.retry_attempted and not pr.refusal
    assert pr.text == "Weak answer [pr_1]."  # answer returned as today
    assert pr.guard_pass is False            # guard reports still attached


def test_empty_retrieval_short_circuits(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_ADAPTIVE_RETRY", True)
    r = FakeRetriever([[]])
    a = FakeAnswerer(["should not be called"])
    nli = FakeNLI([True])
    pr = _answer(r, a, nli)
    assert pr.empty and len(r.calls) == 1
    assert a.calls == 0                      # generation skipped on empty


def test_retry_reason_records_trigger(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_ADAPTIVE_RETRY", True)
    # First answer cites a fabricated id -> reference validator fails it.
    r = FakeRetriever([CHUNKS, CHUNKS])
    a = FakeAnswerer(["Fixed by [pr_999].", "Fixed by [pr_1]."])
    nli = FakeNLI([True, True])
    pr = _answer(r, a, nli)
    assert pr.retry_attempted
    assert "fabricated_citations" in pr.retry_reason
