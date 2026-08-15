"""Phase E: export a failure gallery — an honest record of where RepoMind fails.

Reads the Phase A per-query metrics log and/or a Phase C/D evaluation
``results.json`` (joined with its golden set for ground truth) and selects a
spread of failure cases across categories:

  * guard_rejection    — the guard rejected the answer
  * fabricated_citation — a cited chunk_id was not in the retrieved evidence
  * nli_contradiction   — a claim contradicted its cited evidence
  * retrieval_miss      — gold evidence was absent from the retrieved set
  * incorrect_refusal   — the system abstained on an *answerable* question

Emits ``results/failure_gallery.md`` with one section per case (question, what
the system returned, the ground truth, which stage failed, and a blank
``**Why:**`` line to fill in by hand). Standard library only.

    python scripts/export_failure_gallery.py \
        --metrics data/metrics/queries.jsonl \
        --results results/<run>/results.json \
        --dataset eval/datasets/<slug>.jsonl \
        --out results/failure_gallery.md --limit 15
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.golden_set import load_golden_set          # noqa: E402
from eval.run import looks_like_abstention           # noqa: E402

STAGE = {
    "retrieval_miss": "retrieval",
    "fabricated_citation": "reference validator",
    "nli_contradiction": "NLI verifier",
    "guard_rejection": "guard",
    "incorrect_refusal": "guard / abstention",
}


@dataclass
class FailureCase:
    category: str
    question: str
    returned: str
    ground_truth: str
    stage: str
    source: str


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    if not path or not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def from_metrics(path: Path) -> list[FailureCase]:
    cases = []
    for r in _read_jsonl(path):
        if r.get("empty") or r.get("guard_verdict") != "fail":
            continue
        reason = r.get("guard_reason", "") or ""
        if "fabricated_citations" in reason:
            cat = "fabricated_citation"
        elif r.get("num_contradicted", 0) or "contradicted" in reason:
            cat = "nli_contradiction"
        else:
            cat = "guard_rejection"
        cases.append(FailureCase(
            category=cat, question=r.get("question", ""),
            returned=f"(guard {r.get('guard_verdict')}: {reason})",
            ground_truth="(live query — no ground truth)",
            stage=STAGE[cat], source="metrics"))
    return cases


def from_results(results_path: Path, dataset_path: Path) -> list[FailureCase]:
    if not results_path or not results_path.exists():
        return []
    data = json.loads(results_path.read_text(encoding="utf-8"))
    by_id = {}
    if dataset_path and dataset_path.exists():
        by_id = {e.id: e for e in load_golden_set(dataset_path)}

    cases = []
    for row in data.get("rows", []):
        entry = by_id.get(row.get("id"))
        question = entry.question if entry else row.get("id", "")
        gt = (entry.ground_truth if entry and entry.ground_truth
              else "(none)")
        answer = row.get("answer", "")
        qt = row.get("query_type", "")
        if qt == "unanswerable":
            continue  # abstention here is the *correct* outcome
        # Retrieval miss: gold evidence never retrieved.
        if row.get("recall_at_k") == 0.0:
            cases.append(FailureCase(
                "retrieval_miss", question, answer, gt,
                STAGE["retrieval_miss"], "eval"))
        # Incorrect refusal: abstained on an answerable question.
        abstained = (row.get("refusal") or not row.get("guard_pass", True)
                     or looks_like_abstention(answer))
        if abstained:
            cases.append(FailureCase(
                "incorrect_refusal", question, answer, gt,
                STAGE["incorrect_refusal"], "eval"))
    return cases


def select_spread(cases: list[FailureCase], limit: int) -> list[FailureCase]:
    """Round-robin across categories so the gallery isn't 15 of one bug."""
    # Dedupe by (category, question).
    seen, unique = set(), []
    for c in cases:
        key = (c.category, c.question)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    buckets: dict[str, list[FailureCase]] = {}
    for c in unique:
        buckets.setdefault(c.category, []).append(c)
    picked: list[FailureCase] = []
    while len(picked) < limit and any(buckets.values()):
        for cat in list(buckets):
            if buckets[cat]:
                picked.append(buckets[cat].pop(0))
                if len(picked) >= limit:
                    break
    return picked


def render(cases: list[FailureCase]) -> str:
    lines = ["# RepoMind — failure gallery", "",
             f"{len(cases)} case(s), spread across failure categories. Each records "
             "the stage that failed, what the system returned, and the ground truth.",
             ""]
    if not cases:
        lines.append("_No failure cases found in the supplied sources._")
        return "\n".join(lines)
    for i, c in enumerate(cases, 1):
        lines += [
            f"## {i}. [{c.category}] {c.question or '(no question)'}",
            f"- **Stage that failed:** {c.stage}  (source: {c.source})",
            f"- **System returned:** {c.returned[:500]}",
            f"- **Ground truth:** {c.ground_truth[:500]}",
            "",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Export a RepoMind failure gallery")
    p.add_argument("--metrics", default="data/metrics/queries.jsonl")
    p.add_argument("--results", default="")
    p.add_argument("--dataset", default="")
    p.add_argument("--out", default="results/failure_gallery.md")
    p.add_argument("--limit", type=int, default=15)
    args = p.parse_args(argv)

    cases = from_metrics(Path(args.metrics)) + from_results(
        Path(args.results) if args.results else None,
        Path(args.dataset) if args.dataset else None)
    picked = select_spread(cases, args.limit)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(picked), encoding="utf-8")
    print(f"Wrote {out} ({len(picked)} of {len(cases)} candidate case(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
