"""Evaluation metrics.

Retrieval:   recall@k, MRR, nDCG@k (binary relevance vs gold evidence).
Citations:   precision / recall of an answer's citations vs gold evidence.
Generation:  faithfulness + answer relevancy via a *swappable* judge.

The judge is chosen by a single config variable (``JUDGE_PROVIDER``): Groq or
Gemini (free-tier APIs) or a fully-local Ollama model — never hardcoded to one
provider. It is one config-driven function, not two code paths.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Callable, Iterable

import requests

import config


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _base(chunk_id: str) -> str:
    return chunk_id.split("#", 1)[0]


def _norm(ids: Iterable[str]) -> set[str]:
    return {_base(i) for i in ids}


# --------------------------------------------------------------------------- #
# Retrieval metrics
# --------------------------------------------------------------------------- #
def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    rel = _norm(relevant)
    if not rel:
        return 0.0
    top = _norm(retrieved[:k])
    return len(top & rel) / len(rel)


def mrr(retrieved: list[str], relevant: list[str]) -> float:
    rel = _norm(relevant)
    for i, cid in enumerate(retrieved, start=1):
        if _base(cid) in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    rel = _norm(relevant)
    if not rel:
        return 0.0
    dcg = 0.0
    for i, cid in enumerate(retrieved[:k], start=1):
        if _base(cid) in rel:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


# --------------------------------------------------------------------------- #
# Citation metrics
# --------------------------------------------------------------------------- #
def citation_precision_recall(
    cited: list[str], relevant: list[str]
) -> tuple[float, float]:
    """Precision and recall of an answer's citations vs gold evidence."""
    c = _norm(cited)
    rel = _norm(relevant)
    if not c:
        return (0.0, 0.0 if rel else 1.0)
    hit = len(c & rel)
    precision = hit / len(c)
    recall = hit / len(rel) if rel else 0.0
    return precision, recall


# --------------------------------------------------------------------------- #
# Swappable judge for faithfulness / answer relevancy
# --------------------------------------------------------------------------- #
_JUDGE_PROMPT = """\
You are grading an answer strictly against the provided EVIDENCE.

Return a compact JSON object with two numbers in [0,1]:
  "faithfulness": fraction of the answer's factual claims that are supported by
                  the EVIDENCE (1.0 = fully supported, 0.0 = unsupported/made up),
  "answer_relevancy": how directly the answer addresses the QUESTION.

Respond with ONLY the JSON object, nothing else.

QUESTION:
{question}

EVIDENCE:
{contexts}

ANSWER:
{answer}
"""


def _extract_json(text: str) -> dict[str, Any]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def _judge_via_openai_compatible(prompt: str, base_url: str, api_key: str,
                                 model: str) -> str:
    """Call any OpenAI-compatible chat endpoint (Groq)."""
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return resp.choices[0].message.content or ""


def _judge_via_ollama(prompt: str, model: str) -> str:
    resp = requests.post(
        f"{config.OLLAMA_HOST}/api/chat",
        json={"model": model, "stream": False,
              "messages": [{"role": "user", "content": prompt}],
              "options": {"temperature": 0.0}},
        timeout=config.HTTP_TIMEOUT_SECONDS * 8,
    )
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "")


def _judge_via_gemini(prompt: str, model: str) -> str:
    # OpenAI-compatible Gemini endpoint.
    from openai import OpenAI

    client = OpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=config.GEMINI_API_KEY,
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return resp.choices[0].message.content or ""


def make_judge() -> Callable[[str], str]:
    """Return a text->text judge callable for the configured provider.

    Config-driven via ``JUDGE_PROVIDER``, which may name any provider in
    :mod:`providers` (groq, gemini, nvidia, openrouter, ollama). Falls back to
    the local model when the chosen provider has no key or fails, so a run
    never dies because a free tier ran out.
    """
    import providers

    return lambda prompt: providers.chat(
        config.JUDGE_PROVIDER, prompt, temperature=0.0)[0]


def build_judge_prompt(question: str, answer: str, contexts: list[str]) -> str:
    """The exact judge prompt for a (question, answer, contexts) triple.

    Exposed so callers (e.g. the evaluation runner) can hash it for caching.
    """
    return _JUDGE_PROMPT.format(
        question=question,
        contexts="\n---\n".join(contexts)[:6000],
        answer=answer,
    )


def judge_faithfulness_relevancy(
    question: str, answer: str, contexts: list[str], judge_fn=None,
) -> dict[str, Any]:
    """Score faithfulness + answer relevancy with the swappable judge.

    ``judge_fn`` optionally overrides the text->text judge (used to inject a
    disk-cached, rate-limited judge in the evaluation runner). When None, the
    provider-selected judge from :func:`make_judge` is used.
    """
    # The label reports the provider that will ACTUALLY answer (resolve() drops
    # back to local when a key is missing), so results never claim a provider
    # that was merely requested.
    import providers

    label = providers.describe(config.JUDGE_PROVIDER)

    judge = judge_fn or make_judge()
    prompt = _JUDGE_PROMPT.format(
        question=question,
        contexts="\n---\n".join(contexts)[:6000],
        answer=answer,
    )
    try:
        raw = judge(prompt)
        parsed = _extract_json(raw)
    except Exception as exc:  # noqa: BLE001
        return {"faithfulness": None, "answer_relevancy": None,
                "judge": f"{label} (error: {exc})"}
    return {
        "faithfulness": _clamp01(parsed.get("faithfulness")),
        "answer_relevancy": _clamp01(parsed.get("answer_relevancy")),
        "judge": label,
    }


def _clamp01(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, f))
