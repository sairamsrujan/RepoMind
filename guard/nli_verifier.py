"""Natural Language Inference (NLI) hallucination guard.

For each claim (sentence) in an answer and each chunk it cites, run NLI between
the cited chunk's text (premise) and the claim (hypothesis). A claim is:
  * ENTAILED   if some cited chunk entails it (prob >= threshold),
  * CONTRADICTED if some cited chunk contradicts it (and that dominates),
  * UNVERIFIED otherwise (cited evidence neither entails nor contradicts).

Contradicted or unverified claims are surfaced so the UI can warn the user.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import config

# One definition, shared with the reference validator — the two guard stages
# must agree on what a citation is, or one flags what the other ignores.
from generation.answerer import _CITATION_RE, normalise_citation_brackets  # noqa: E402
_MODEL_CACHE: dict[str, Any] = {}


def _base_id(chunk_id: str) -> str:
    return chunk_id.split("#", 1)[0]


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def split_claims(text: str) -> list[str]:
    """Split an answer into claim-sized sentences (keeps citation markers)."""
    # Split on sentence boundaries and newlines/bullets.
    rough = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    claims = []
    for s in rough:
        s = s.strip().lstrip("-*• ").strip()
        if len(s) >= 3:
            claims.append(s)
    return claims


@dataclass
class ClaimVerdict:
    claim: str
    citations: list[str]
    entailment: float
    contradiction: float
    status: str  # "entailed" | "contradicted" | "unverified" | "uncited"


@dataclass
class NLIReport:
    claims: list[ClaimVerdict] = field(default_factory=list)

    @property
    def contradicted(self) -> list[ClaimVerdict]:
        return [c for c in self.claims if c.status == "contradicted"]

    @property
    def unverified(self) -> list[ClaimVerdict]:
        return [c for c in self.claims if c.status == "unverified"]

    @property
    def is_grounded(self) -> bool:
        """No contradicted claims among those that carry citations."""
        return not self.contradicted


class NLIVerifier:
    """Cross-encoder NLI verifier (contradiction / entailment / neutral)."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.NLI_MODEL
        self._model = None
        self._labels: dict[int, str] = {0: "contradiction", 1: "entailment",
                                        2: "neutral"}

    def _load(self):
        if self._model is not None:
            return self._model
        if self.model_name in _MODEL_CACHE:
            self._model = _MODEL_CACHE[self.model_name]
        else:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, max_length=512,
                                       device=config.TORCH_DEVICE)
            _MODEL_CACHE[self.model_name] = self._model
        # Read the model's own label mapping when available.
        cfg = getattr(getattr(self._model, "model", None), "config", None)
        if cfg is not None and getattr(cfg, "id2label", None):
            self._labels = {int(k): str(v).lower() for k, v in cfg.id2label.items()}
        return self._model

    def _to_probs(self, logits) -> dict[str, float]:
        probs = _softmax(np.asarray(logits, dtype=float))
        out = {"contradiction": 0.0, "entailment": 0.0, "neutral": 0.0}
        for idx, p in enumerate(probs):
            label = self._labels.get(idx, str(idx))
            for key in out:
                if key in label:
                    out[key] = float(p)
        return out

    def predict_nli(self, premise: str, hypothesis: str) -> dict[str, float]:
        return self.predict_nli_batch([(premise, hypothesis)])[0]

    def predict_nli_batch(
        self, pairs: list[tuple[str, str]]
    ) -> list[dict[str, float]]:
        """Score many (premise, hypothesis) pairs in ONE forward pass.

        The guard checks each claim against its combined evidence *and* each
        cited chunk individually. Doing those as separate single-item calls
        costs one model invocation each, which dominates query latency on CPU;
        batching them is the same computation in a fraction of the time.
        """
        if not pairs:
            return []
        model = self._load()
        raw = model.predict(pairs)
        return [self._to_probs(row) for row in raw]

    def verify(
        self, answer_text: str, retrieved_chunks: list[dict[str, Any]]
    ) -> NLIReport:
        text_by_id: dict[str, str] = {}
        for c in retrieved_chunks:
            text_by_id[c["chunk_id"]] = c.get("text", "")
            text_by_id.setdefault(_base_id(c["chunk_id"]), c.get("text", ""))

        report = NLIReport()
        # Same normalisation the reference validator applies, so both guard
        # stages see the same citations in the same answer.
        for claim in split_claims(normalise_citation_brackets(answer_text)):
            citations = _CITATION_RE.findall(claim)
            hypothesis = _CITATION_RE.sub("", claim).strip()
            if not citations:
                # Uncited claims aren't NLI-checkable here (reference_validator
                # separately flags a fully-uncited answer).
                report.claims.append(ClaimVerdict(claim, [], 0.0, 0.0, "uncited"))
                continue

            # NLI against the UNION of the claim's cited evidence, plus each
            # chunk individually. entailment = the best support from any view;
            # contradiction = only when the combined evidence contradicts. This
            # avoids falsely flagging a well-supported claim just because one
            # differently-phrased sibling citation (e.g. an issue written as a
            # question) looks contradictory in isolation.
            premises = []
            for cid in citations:
                p = text_by_id.get(cid) or text_by_id.get(_base_id(cid))
                if p:
                    premises.append(p)
            if not premises:
                report.claims.append(
                    ClaimVerdict(claim, citations, 0.0, 0.0, "unverified"))
                continue

            combined = "\n".join(premises)[:4000]
            # One batched forward pass for the combined premise plus each
            # individual one, instead of len(premises)+1 separate calls.
            scored = self.predict_nli_batch(
                [(combined, hypothesis)] + [(p, hypothesis) for p in premises])
            best_ent = max(r["entailment"] for r in scored)
            best_con = scored[0]["contradiction"]    # combined evidence only

            if best_ent >= config.NLI_ENTAILMENT_THRESHOLD:
                status = "entailed"
            elif best_con >= config.NLI_CONTRADICTION_THRESHOLD:
                status = "contradicted"
            else:
                status = "unverified"
            report.claims.append(
                ClaimVerdict(claim, citations, best_ent, best_con, status)
            )
        return report
