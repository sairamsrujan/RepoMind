"""Ablation study: what each pipeline stage actually contributes.

Eight configurations (``ABLATION_CONFIGS``), building up the pipeline stage by
stage and then isolating each retrieval channel:

  1. retrieval-only      RRF hybrid, no MMR, no reranker
  2. + MMR
  3. + MMR + reranker
  4. full + guard        the production configuration
  5. + adaptive retry
  6. + graph expansion
  dense-only / sparse-only   which retrieval channel is carrying the result

Reports recall@k, MRR, nDCG, citation precision/recall, faithfulness and
latency per configuration and per query_type, to ``ablation.csv``/``.json``.

Two things make the output trustworthy, both learned the hard way:

* **Generation is pinned to one model.** Configurations run in order and a cloud
  quota depletes in order, so leaving generation on the fallback chain makes the
  answering model degrade down the table and confounds every comparison.
* **Progress is checkpointed per configuration**, so an interruption costs one
  configuration rather than the whole multi-hour run.

Runs offline against already-indexed repositories — never in the live query path.
"""
from __future__ import annotations

import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import config
from core.context import RepositoryContext
from eval import metrics
from eval.synth_questions import GoldItem, synthesize_gold_set
from generation.answerer import Answerer
from guard.nli_verifier import NLIVerifier
from guard.reference_validator import validate_references
from retrieval.retriever import Retriever

CONFIGS = [
    {"name": "1. retrieval-only", "mmr": False, "reranker": False, "guard": False},
    {"name": "2. + MMR", "mmr": True, "reranker": False, "guard": False},
    {"name": "3. + MMR + reranker", "mmr": True, "reranker": True, "guard": False},
    {"name": "4. full + guard", "mmr": True, "reranker": True, "guard": True},
]


@dataclass
class ConfigResult:
    name: str
    faithfulness: float | None
    answer_relevancy: float | None
    citation_precision: float
    recall_at_k: float
    avg_latency_s: float
    guard_flags: int = 0
    n: int = 0
    judge: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def _mean(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else 0.0


def run_config(
    retriever: Retriever,
    answerer: Answerer,
    nli: NLIVerifier | None,
    gold: list[GoldItem],
    cfg: dict,
    coverage: tuple[str, str],
    judge_faithfulness: bool = True,
) -> ConfigResult:
    since, until = coverage
    faiths, rels, cprecs, recalls, lats = [], [], [], [], []
    guard_flags = 0
    judge_label = ""

    for item in gold:
        t0 = time.time()
        chunks = retriever.retrieve(
            item.question, use_mmr=cfg["mmr"], use_reranker=cfg["reranker"],
        )
        result = answerer.answer(item.question, chunks, since, until)
        lats.append(time.time() - t0)

        retrieved_ids = [c["chunk_id"] for c in chunks]
        recalls.append(metrics.recall_at_k(
            retrieved_ids, item.evidence_chunk_ids, config.FINAL_TOP_K))
        cp, _ = metrics.citation_precision_recall(
            result.cited_chunk_ids, item.evidence_chunk_ids)
        cprecs.append(cp)

        if judge_faithfulness:
            scored = metrics.judge_faithfulness_relevancy(
                item.question, result.text, [c["text"] for c in chunks])
            faiths.append(scored["faithfulness"])
            rels.append(scored["answer_relevancy"])
            judge_label = scored["judge"]

        if cfg["guard"]:
            ref_rep = validate_references(result.text, chunks)
            guard_flags += len(ref_rep.invalid_citations)
            if nli is not None:
                nli_rep = nli.verify(result.text, chunks)
                guard_flags += len(nli_rep.contradicted)

    return ConfigResult(
        name=cfg["name"],
        faithfulness=_mean(faiths) if judge_faithfulness else None,
        answer_relevancy=_mean(rels) if judge_faithfulness else None,
        citation_precision=_mean(cprecs),
        recall_at_k=_mean(recalls),
        avg_latency_s=_mean(lats),
        guard_flags=guard_flags,
        n=len(gold),
        judge=judge_label,
    )


def run_ablation(
    ctx: RepositoryContext,
    gold: list[GoldItem] | None = None,
    max_questions: int | None = None,
    judge_faithfulness: bool = True,
) -> list[ConfigResult]:
    """Run all four configurations over ``ctx``'s gold set."""
    gold = gold if gold is not None else synthesize_gold_set(ctx)
    if max_questions:
        gold = gold[:max_questions]
    if not gold:
        return []

    from core import manifest as manifest_mod

    m = manifest_mod.read_manifest(ctx.manifest_path)
    coverage = (m["coverage"].get("since", ""), m["coverage"].get("until", ""))

    retriever = Retriever(ctx)
    answerer = Answerer(use_chain=True)   # offline: prefer cloud over speed
    nli = NLIVerifier()

    results = []
    for cfg in CONFIGS:
        results.append(run_config(
            retriever, answerer, nli, gold, cfg, coverage,
            judge_faithfulness=judge_faithfulness,
        ))
    return results


def format_table(results: list[ConfigResult], title: str = "") -> str:
    """Render the ablation comparison as a fixed-width table."""
    if not results:
        return f"{title}\n(no gold questions — need closed issues resolved by " \
               f"merged PRs)\n"
    header = (f"{'Configuration':<22}{'Faithful':>10}{'Relevancy':>11}"
              f"{'Cite-Prec':>11}{'Recall@k':>10}{'Latency(s)':>12}{'Guard⚑':>8}")
    lines = [title, "=" * len(header), header, "-" * len(header)]
    for r in results:
        f = f"{r.faithfulness:.3f}" if r.faithfulness is not None else "  n/a"
        rel = f"{r.answer_relevancy:.3f}" if r.answer_relevancy is not None else "  n/a"
        lines.append(
            f"{r.name:<22}{f:>10}{rel:>11}{r.citation_precision:>11.3f}"
            f"{r.recall_at_k:>10.3f}{r.avg_latency_s:>12.2f}{r.guard_flags:>8}"
        )
    if results[0].judge:
        lines.append("-" * len(header))
        lines.append(f"judge: {results[0].judge}   n={results[0].n} questions")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Phase D: golden-set ablation with channel isolation + adaptive retry, broken
# down by query_type, with machine-readable CSV/JSON output and multi-repo runs.
# --------------------------------------------------------------------------- #
# Each config specifies a retrieval/verification strategy. The first four match
# the classic ablation; #5 adds the adaptive retry; the last two isolate the
# dense and sparse channels so the BM25 channel's contribution is *measured*.
ABLATION_CONFIGS = [
    {"name": "1.retrieval-only", "mmr": False, "reranker": False,
     "dense": True, "sparse": True, "retry": False, "graph": False},
    {"name": "2.+MMR", "mmr": True, "reranker": False,
     "dense": True, "sparse": True, "retry": False, "graph": False},
    {"name": "3.+MMR+reranker", "mmr": True, "reranker": True,
     "dense": True, "sparse": True, "retry": False, "graph": False},
    {"name": "4.full+guard", "mmr": True, "reranker": True,
     "dense": True, "sparse": True, "retry": False, "graph": False},
    {"name": "5.full+guard+retry", "mmr": True, "reranker": True,
     "dense": True, "sparse": True, "retry": True, "graph": False},
    # Graph expansion is measured, not assumed: a retrieval-only A/B showed it
    # raises recall but costs latency and slightly lowers nDCG, so whether it
    # earns its place is exactly the kind of question an ablation should answer.
    {"name": "6.full+graph", "mmr": True, "reranker": True,
     "dense": True, "sparse": True, "retry": False, "graph": True},
    {"name": "dense-only", "mmr": True, "reranker": True,
     "dense": True, "sparse": False, "retry": False, "graph": False},
    {"name": "sparse-only", "mmr": True, "reranker": True,
     "dense": False, "sparse": True, "retry": False, "graph": False},
]

_CSV_METRICS = ["recall_at_k", "mrr", "ndcg", "citation_precision",
                "citation_recall", "faithfulness", "answer_relevancy",
                "abstention_accuracy", "latency_ms_mean", "latency_ms_p95"]


def _result_for(query_pipeline, retriever, answerer, nli, question, since, until,
                cfg):
    """Produce a scored-answer object for one config (retry or single attempt)."""
    if cfg["retry"]:
        old = config.ENABLE_ADAPTIVE_RETRY
        config.ENABLE_ADAPTIVE_RETRY = True
        try:
            return query_pipeline.answer_query(retriever, answerer, nli,
                                               question, since, until)
        finally:
            config.ENABLE_ADAPTIVE_RETRY = old
    return query_pipeline.run_attempt(
        retriever, answerer, nli, question, since, until,
        use_mmr=cfg["mmr"], use_reranker=cfg["reranker"],
        use_dense=cfg["dense"], use_sparse=cfg["sparse"],
        use_graph=cfg.get("graph", False))


def run_golden_ablation(ctx, entries, metrics_list, judge, judge_state,
                        configs=None, checkpoint_dir=None) -> dict:
    """Run every configuration over ``entries``; return per-config breakdowns.

    Cost is configurations x questions, so this runs for hours. It is
    checkpointed per configuration: each completed config is written to
    ``checkpoint_dir`` immediately and skipped on a re-run. Without this an
    interruption near the end — a dropped Ollama, a closed laptop, an OOM kill —
    discarded the entire run, which is exactly what happened before it was added.
    """
    import json as _json
    import query_pipeline

    from core import manifest as manifest_mod
    from eval import run as eval_run

    configs = configs or ABLATION_CONFIGS
    m = manifest_mod.read_manifest(ctx.manifest_path)
    since = m["coverage"].get("since", "")
    until = m["coverage"].get("until", "")

    # The retriever only builds its neighbour map when graph expansion is
    # enabled at construction time. Force it on if ANY config needs it, and let
    # the per-call ``use_graph`` flag decide config by config — otherwise the
    # graph configuration would silently measure the non-graph pipeline.
    _flag = config.ENABLE_GRAPH_EXPANSION
    if any(c.get("graph") for c in configs):
        config.ENABLE_GRAPH_EXPANSION = True
    try:
        retriever = Retriever(ctx)
    finally:
        config.ENABLE_GRAPH_EXPANSION = _flag
    # An ablation isolates ONE variable: the configuration. The generation model
    # must therefore be identical for every configuration.
    #
    # This is not hypothetical. A previous run left generation on the normal
    # cloud-with-fallback path, and because configurations execute in order
    # while the daily quota depletes in order, the answering model degraded
    # monotonically down the table:
    #
    #   1.retrieval-only   19/20 answers from Groq-70B
    #   3.+MMR+reranker     0/20 answers from Groq-70B (13 × 49B, 7 × local 7B)
    #
    # The resulting "faithfulness falls as you add pipeline stages" was mostly
    # the answerer getting weaker, not the stages hurting. Consistency matters
    # more here than raw model quality, so generation is pinned to a single
    # model — local by default, since that is the one model always available and
    # immune to quota depletion mid-run.
    answerer, nli = Answerer(use_chain=False), NLIVerifier()

    # Resume: reload any configurations already completed for this repository.
    from pathlib import Path as _Path
    ckpt = None
    if checkpoint_dir:
        ckpt = _Path(checkpoint_dir) / f"partial-{ctx.slug}.json"
        ckpt.parent.mkdir(parents=True, exist_ok=True)

    out: dict = {}
    if ckpt and ckpt.exists():
        try:
            out = _json.loads(ckpt.read_text(encoding="utf-8"))
            if out:
                print(f"  resuming: {len(out)} config(s) already done "
                      f"({', '.join(out)})", file=sys.stderr)
        except (OSError, ValueError):
            out = {}

    for cfg in configs:
        if cfg["name"] in out:
            continue
        print(f"  [{cfg['name']}] {len(entries)} questions", file=sys.stderr)
        rows = []
        for e in entries:
            t0 = time.perf_counter()
            pr = _result_for(query_pipeline, retriever, answerer, nli,
                             e.question, since, until, cfg)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            rows.append(eval_run.metrics_for_entry(
                pr, e, metrics_list, judge, judge_state, latency_ms))
        out[cfg["name"]] = {"by_type": eval_run.aggregate(rows), "rows": rows}
        # Persist after EVERY configuration, not at the end of the run.
        if ckpt:
            try:
                ckpt.write_text(_json.dumps(out), encoding="utf-8")
            except OSError:
                pass          # a failed checkpoint must not abort the run
    return out


def write_ablation_outputs(all_results: dict, out_dir,
                           generation_model: str = "") -> None:
    """Write ablation.csv (per repo/config/query_type) and ablation.json."""
    import csv
    import json
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Count what actually answered, per config. A table whose configurations
    # were answered by different models is comparing models, not configurations,
    # so this must travel with the numbers rather than be reconstructed later.
    answered = {
        repo: {cname: _answered_by(data.get("rows", []))
               for cname, data in configs.items()}
        for repo, configs in all_results.items()
    }
    payload = {
        "generation_model": generation_model,
        "answered_by_per_config": answered,
        "results": all_results,
    }
    (out_dir / "ablation.json").write_text(json.dumps(payload, indent=2),
                                           encoding="utf-8")
    with (out_dir / "ablation.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["repo", "config", "query_type", "n"] + _CSV_METRICS)
        for repo, configs in all_results.items():
            for cname, data in configs.items():
                for qt, s in data["by_type"].items():
                    w.writerow([repo, cname, qt, s.get("n", 0)]
                               + [_fmt(s.get(k)) for k in _CSV_METRICS])


def _answered_by(rows: list) -> dict:
    """Count how many answers each model produced within one configuration."""
    counts: dict = {}
    for r in rows:
        m = r.get("model") or "(unrecorded)"
        counts[m] = counts.get(m, 0) + 1
    return counts


def _fmt(v) -> str:
    return "" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def format_golden_ablation(all_results: dict) -> str:
    lines = []
    for repo, configs in all_results.items():
        lines.append(f"\n=== Ablation — {repo} ===")
        hdr = f"{'config':<20}{'recall@k':>10}{'cite-prec':>10}{'faithful':>10}" \
              f"{'abstain':>9}{'lat.ms':>9}"
        lines += [hdr, "-" * len(hdr)]
        for cname, data in configs.items():
            # Aggregate a couple of headline numbers across answerable + unanswerable.
            answerable = [s for qt, s in data["by_type"].items()
                          if qt != "unanswerable"]
            un = data["by_type"].get("unanswerable")
            rec = _avg([s.get("recall_at_k") for s in answerable])
            cp = _avg([s.get("citation_precision") for s in answerable])
            fa = _avg([s.get("faithfulness") for s in answerable])
            ab = un.get("abstention_accuracy") if un else None
            lat = _avg([s.get("latency_ms_mean") for s in data["by_type"].values()])
            lines.append(f"{cname:<20}{_c(rec):>10}{_c(cp):>10}{_c(fa):>10}"
                         f"{_c(ab):>9}{_c(lat, 0):>9}")
    return "\n".join(lines)


def _avg(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    return statistics.fmean(vals) if vals else None


def _c(v, prec=3) -> str:
    return "n/a" if v is None else f"{v:.{prec}f}"


def main(argv: list[str] | None = None) -> int:
    import argparse
    from datetime import datetime
    from pathlib import Path

    from core.repo_url import parse_repo_url
    from eval import run as eval_run
    from eval.golden_set import load_golden_set, validate_golden_set

    p = argparse.ArgumentParser(description="RepoMind ablation harness")
    p.add_argument("--repos", required=True,
                   help="comma-separated owner/name list (each already indexed)")
    p.add_argument("--datasets", default="",
                   help="comma-separated golden-set paths (defaults to "
                        "eval/datasets/<slug>.jsonl per repo)")
    p.add_argument("--metrics", default=",".join(eval_run.ALL_METRICS))
    p.add_argument("--out", default="")
    p.add_argument("--sleep", type=float, default=0.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--allow-cloud-generation", action="store_true",
                   help="DO NOT USE for a real ablation. Leaves generation on "
                        "the cloud chain, which makes the answering model "
                        "degrade as the quota depletes and confounds every "
                        "configuration comparison in the table.")
    args = p.parse_args(argv)

    # Pin generation to one model unless explicitly overridden. See the comment
    # in run_golden_ablation for the run this guard exists because of.
    if not args.allow_cloud_generation:
        config.GENERATION_PROVIDER = "ollama"
        print(f"generation pinned to {config.GENERATION_MODEL} for every "
              f"configuration (comparability); pass --allow-cloud-generation "
              f"to disable this guard", file=sys.stderr)

    repos = [r.strip() for r in args.repos.split(",") if r.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    metrics_list = [x.strip() for x in args.metrics.split(",") if x.strip()]
    out_dir = Path(args.out) if args.out else (
        Path("results") / datetime.now().strftime("ablation-%Y%m%d-%H%M%S"))
    generation_model = (config.effective_generation_model()
                        if args.allow_cloud_generation else config.GENERATION_MODEL)
    judge = eval_run.CachedJudge(out_dir / "judge_cache.json", args.sleep)
    judge_state: dict = {"available": True}

    all_results: dict = {}
    for i, repo in enumerate(repos):
        ref = parse_repo_url(repo)
        ctx = RepositoryContext.for_ref(ref)
        if not ctx.manifest_path.exists():
            print(f"skip {ref.full_name}: not indexed", file=__import__("sys").stderr)
            continue
        ds = datasets[i] if i < len(datasets) else \
            f"eval/datasets/{ref.slug}.jsonl"
        entries = load_golden_set(ds)
        validate_golden_set(entries)
        if args.limit:
            entries = entries[:args.limit]
        print(f"Running ablation for {ref.full_name} ({len(entries)} questions)…",
              file=__import__("sys").stderr)
        all_results[ref.full_name] = run_golden_ablation(
            ctx, entries, metrics_list, judge, judge_state,
            checkpoint_dir=out_dir)

    if not all_results:
        print("No indexed repositories to ablate.", file=__import__("sys").stderr)
        return 1
    write_ablation_outputs(all_results, out_dir, generation_model)
    print(format_golden_ablation(all_results))
    if not judge_state.get("available", True):
        print(f"\nevaluation unavailable (judge): {judge_state.get('reason')}")
    print(f"\nWrote {out_dir}/ablation.csv and ablation.json",
          file=__import__("sys").stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
