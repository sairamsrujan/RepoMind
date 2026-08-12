"""Phase B: the answer pipeline with an optional adaptive verification retry.

This is plain, deterministic Python — no orchestration framework. It ties
together the existing retriever, answerer, and hallucination guard, and adds one
feedback step: if the guard rejects the answer and
``config.ENABLE_ADAPTIVE_RETRY`` is on, it retries **exactly once** with a
widened retrieval strategy (bigger dense+BM25 pools, MMR skipped so the
highest-scoring evidence is kept, reranker still on). If the retry is still
rejected, it returns an honest refusal rather than an unverified answer.

Behaviour with ``ENABLE_ADAPTIVE_RETRY = False`` (the default) is identical to
before: one attempt, whose answer is returned exactly as today (guard reports
are still attached for the UI pills). Retries are hard-bounded at one — no
recursion, no loops.

Import-safe for the live path: imports only ``config``, ``guard/``, and the
standard library. It never imports from ``eval/``.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

import config
from guard.reference_validator import validate_references

REFUSAL_TEXT = (
    "I could not find sufficient evidence in the indexed history to answer this "
    "question with confidence. The retrieved material below did not support a "
    "verifiable answer, so rather than guess I'm declining to answer. You can "
    "see what was retrieved under “Evidence used”, try rephrasing the "
    "question, or widen the indexed date range."
)


@dataclass
class Attempt:
    chunks: list[dict[str, Any]]
    text: str
    ref_report: Any
    nli_report: Any
    guard_pass: bool
    generation_ms: float
    guard_ms: float
    empty: bool = False
    model: str = ""              # model that actually produced the text
    fell_back: bool = False      # cloud generation failed -> local answered
    fallback_reason: str = ""


@dataclass
class PipelineResult:
    empty: bool
    chunks: list[dict[str, Any]]
    text: str                       # what to DISPLAY (answer or refusal)
    ref_report: Any                 # guard report for the model's final answer
    nli_report: Any
    guard_pass: bool
    generation_ms: float
    guard_ms: float
    model: str = ""              # model that actually produced the text
    fell_back: bool = False      # cloud generation failed -> local answered
    fallback_reason: str = ""
    retry_attempted: bool = False
    retry_reason: str = ""
    retry_succeeded: bool = False
    refusal: bool = False        # guard rejected, and the retry also failed
    declined: bool = False       # the model itself found nothing to answer with


# A model that correctly finds nothing writes "there is no evidence…" and cites
# the chunks it looked at. Those citations are real and the statement is
# entailed, so the guard passes it — and the UI then showed green "verified"
# badges on what is actually a non-answer, which reads as success. Detecting
# this lets the interface say "nothing found" instead of implying it answered.
#
# Lives here, not in eval/, because the live app must not import from eval/
# (see the architectural rule in README). eval/run.py imports THIS.
_DECLINED_RE = re.compile(
    r"\b(do(?:es)?\s+not\s+(?:contain|cover|mention|include)|insufficient|"
    r"cannot\s+(?:answer|provide|find|determine)|not\s+covered|outside\s+the\s+"
    r"indexed|no\s+(?:information|evidence)|declin\w+\s+to\s+answer|"
    r"unable\s+to\s+answer)\b", re.IGNORECASE)


def looks_like_declination(text: str) -> bool:
    """True when an answer is really a statement that nothing was found."""
    return bool(_DECLINED_RE.search(text or ""))


def guard_reason(ref_report, nli_report) -> str:
    """Short human-readable reason a guard verdict failed (or 'ok')."""
    parts = []
    if ref_report is not None and ref_report.invalid_citations:
        parts.append(f"fabricated_citations={len(ref_report.invalid_citations)}")
    if nli_report is not None and nli_report.contradicted:
        parts.append(f"contradicted_claims={len(nli_report.contradicted)}")
    return "; ".join(parts) if parts else "ok"


def run_attempt(
    retriever, answerer, nli_verifier, question: str, since: str, until: str,
    *, use_mmr: bool = True, use_reranker: bool = True, use_dense: bool = True,
    use_sparse: bool = True, use_graph: bool = True, dense_k: int | None = None,
    sparse_k: int | None = None, trace: dict | None = None,
) -> Attempt:
    """One full retrieve → generate → guard pass. No retry logic here.

    The ``use_*`` flags let the offline ablation isolate retrieval channels /
    stages; all default to the live pipeline's values, so callers that omit them
    (e.g. :func:`answer_query`) behave exactly as before.
    """
    chunks = retriever.retrieve(
        question, use_mmr=use_mmr, use_reranker=use_reranker, use_dense=use_dense,
        use_sparse=use_sparse, use_graph=use_graph, dense_k=dense_k,
        sparse_k=sparse_k, trace=trace,
    )
    if not chunks:
        return Attempt(chunks=[], text="", ref_report=None, nli_report=None,
                       guard_pass=False, generation_ms=0.0, guard_ms=0.0,
                       empty=True)

    _g0 = time.perf_counter()
    result = answerer.answer(question, chunks, since, until)
    generation_ms = (time.perf_counter() - _g0) * 1000.0

    _guard0 = time.perf_counter()
    ref_report = validate_references(result.text, chunks)
    nli_report = nli_verifier.verify(result.text, chunks)
    guard_ms = (time.perf_counter() - _guard0) * 1000.0

    guard_pass = ref_report.is_valid and nli_report.is_grounded
    return Attempt(model=getattr(result, "model", ""),
                   fell_back=getattr(result, "fell_back", False),
                   fallback_reason=getattr(result, "fallback_reason", ""),
                   chunks=chunks, text=result.text, ref_report=ref_report,
                   nli_report=nli_report, guard_pass=guard_pass,
                   generation_ms=generation_ms, guard_ms=guard_ms)


def answer_query(
    retriever, answerer, nli_verifier, question: str, since: str, until: str,
    trace: dict | None = None,
) -> PipelineResult:
    """Answer ``question``, retrying once on guard rejection when enabled."""
    first = run_attempt(retriever, answerer, nli_verifier, question, since, until,
                        use_mmr=True, trace=trace)
    if first.empty:
        return PipelineResult(empty=True, chunks=[], text="", ref_report=None,
                              nli_report=None, guard_pass=False,
                              generation_ms=0.0, guard_ms=0.0)

    # Happy path, or retry disabled: return the first attempt exactly as today.
    if first.guard_pass or not config.ENABLE_ADAPTIVE_RETRY:
        return PipelineResult(
            empty=False, chunks=first.chunks, text=first.text,
            ref_report=first.ref_report, nli_report=first.nli_report,
            guard_pass=first.guard_pass, generation_ms=first.generation_ms,
            guard_ms=first.guard_ms, retry_attempted=False,
            model=first.model, fell_back=first.fell_back,
            fallback_reason=first.fallback_reason,
            declined=looks_like_declination(first.text))

    # Guard rejected AND retry enabled -> exactly ONE widened retry.
    reason = guard_reason(first.ref_report, first.nli_report)
    retry = run_attempt(
        retriever, answerer, nli_verifier, question, since, until,
        use_mmr=False,
        dense_k=config.DENSE_TOP_K * config.RETRY_POOL_MULTIPLIER,
        sparse_k=config.SPARSE_TOP_K * config.RETRY_POOL_MULTIPLIER,
        trace=trace,
    )

    gen_ms = first.generation_ms + retry.generation_ms
    guard_ms = first.guard_ms + retry.guard_ms

    if not retry.empty and retry.guard_pass:
        return PipelineResult(
            empty=False, chunks=retry.chunks, text=retry.text,
            ref_report=retry.ref_report, nli_report=retry.nli_report,
            guard_pass=True, generation_ms=gen_ms, guard_ms=guard_ms,
            retry_attempted=True, retry_reason=reason, retry_succeeded=True,
            model=retry.model, fell_back=retry.fell_back,
            fallback_reason=retry.fallback_reason,
            declined=looks_like_declination(retry.text))

    # Retry also failed (or produced nothing) -> honest refusal. We keep the
    # retry's guard reports (for metrics) but NEVER display its unverified text.
    return PipelineResult(
        empty=False, chunks=retry.chunks or first.chunks, text=REFUSAL_TEXT,
        ref_report=retry.ref_report or first.ref_report,
        nli_report=retry.nli_report or first.nli_report,
        guard_pass=False, generation_ms=gen_ms, guard_ms=guard_ms,
        retry_attempted=True, retry_reason=reason, retry_succeeded=False,
        refusal=True, model=retry.model or first.model,
        fell_back=retry.fell_back or first.fell_back,
        fallback_reason=retry.fallback_reason or first.fallback_reason)


def answer_across_repos(question: str, repos, answerer, nli_verifier):
    """Phase F: answer one question independently against several repositories.

    ``repos`` is an iterable of ``(label, retriever, since, until)``. Each repo
    is retrieved from its own index (they are never merged) and answered +
    guarded independently. Returns a list of ``(label, PipelineResult)`` in the
    given order, so the UI can present the answers side by side with citations
    unambiguously attributed to their source repository.
    """
    results = []
    for label, retriever, since, until in repos:
        pr = answer_query(retriever, answerer, nli_verifier, question,
                          since, until)
        results.append((label, pr))
    return results
