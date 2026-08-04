"""Measure whether graph expansion actually improves retrieval, per category.

Retrieval-only A/B: for every question in a golden set, retrieve with graph
expansion OFF and ON and compare recall@k / MRR / nDCG against the gold
evidence. No generation, no judge, so this costs no API quota and isolates the
retrieval change from everything downstream.

The point is to *measure* the claim rather than assert it — and in particular to
confirm the evolution category improves without the other categories regressing.

    python scripts/measure_graph_expansion.py --repo pallets/click
    python scripts/measure_graph_expansion.py --repo pallets/click --limit 20
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402
from core.context import RepositoryContext  # noqa: E402
from core.repo_url import parse_repo_url  # noqa: E402
from eval import metrics  # noqa: E402
from eval.golden_set import load_golden_set  # noqa: E402
from eval.run import canonical_id  # noqa: E402


def score(retriever, entries, use_graph: bool) -> dict:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        if e.is_unanswerable:
            continue           # no gold evidence to measure recall against
        t0 = time.perf_counter()
        chunks = retriever.retrieve(e.question, use_graph=use_graph)
        ms = (time.perf_counter() - t0) * 1000.0
        got = [canonical_id(c["chunk_id"]) for c in chunks]
        gold = [canonical_id(x) for x in e.evidence]
        by_cat[e.query_type].append({
            "recall": metrics.recall_at_k(got, gold, config.FINAL_TOP_K),
            "mrr": metrics.mrr(got, gold),
            "ndcg": metrics.ndcg_at_k(got, gold, config.FINAL_TOP_K),
            "ms": ms,
        })
    return by_cat


def _mean(rows, key):
    vals = [r[key] for r in rows]
    return statistics.fmean(vals) if vals else 0.0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="A/B test graph expansion")
    p.add_argument("--repo", required=True)
    p.add_argument("--dataset", default="")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    ref = parse_repo_url(args.repo)
    ctx = RepositoryContext.for_ref(ref)
    dataset = Path(args.dataset) if args.dataset else \
        _ROOT / "eval" / "datasets" / f"{ref.slug}.jsonl"
    entries = load_golden_set(dataset)
    if args.limit:
        entries = entries[:args.limit]

    if not ctx.links_path.exists():
        print(f"{ref.full_name} has no links.json — nothing to expand.",
              file=sys.stderr)
        return 1

    # The retriever only builds its neighbour map when the flag is on at
    # construction time, so force it on and toggle per-call instead.
    config.ENABLE_GRAPH_EXPANSION = True
    from retrieval.retriever import Retriever
    retriever = Retriever(ctx)
    if not retriever._neighbours:
        print("neighbour map is empty — is links.json populated?", file=sys.stderr)
        return 1

    print(f"Graph expansion A/B — {ref.full_name}  "
          f"({len([e for e in entries if not e.is_unanswerable])} answerable "
          f"questions, {len(retriever._neighbours)} linked records)\n")

    off = score(retriever, entries, use_graph=False)
    on = score(retriever, entries, use_graph=True)

    hdr = (f"{'category':<14}{'n':>4}{'recall OFF':>12}{'recall ON':>11}"
           f"{'delta':>9}{'nDCG OFF':>10}{'nDCG ON':>9}{'ms OFF':>9}{'ms ON':>8}")
    print(hdr)
    print("-" * len(hdr))
    all_off, all_on = [], []
    for cat in sorted(off):
        a, b = off[cat], on[cat]
        ra, rb = _mean(a, "recall"), _mean(b, "recall")
        all_off += a
        all_on += b
        arrow = "  " if abs(rb - ra) < 1e-9 else (" +" if rb > ra else " ")
        print(f"{cat:<14}{len(a):>4}{ra:>12.3f}{rb:>11.3f}{arrow}{rb-ra:>7.3f}"
              f"{_mean(a,'ndcg'):>10.3f}{_mean(b,'ndcg'):>9.3f}"
              f"{_mean(a,'ms'):>9.0f}{_mean(b,'ms'):>8.0f}")
    print("-" * len(hdr))
    ra, rb = _mean(all_off, "recall"), _mean(all_on, "recall")
    print(f"{'OVERALL':<14}{len(all_off):>4}{ra:>12.3f}{rb:>11.3f}  {rb-ra:>7.3f}"
          f"{_mean(all_off,'ndcg'):>10.3f}{_mean(all_on,'ndcg'):>9.3f}"
          f"{_mean(all_off,'ms'):>9.0f}{_mean(all_on,'ms'):>8.0f}")

    print(f"\nverdict: recall {'improved' if rb > ra else 'did NOT improve'} "
          f"({ra:.3f} -> {rb:.3f}), "
          f"latency {_mean(all_off,'ms'):.0f}ms -> {_mean(all_on,'ms'):.0f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
