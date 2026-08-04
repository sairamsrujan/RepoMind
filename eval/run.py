"""Phase C2: the evaluation runner.

Queries the live RepoMind pipeline for each golden-set question, computes the
requested metrics (reusing ``eval/metrics.py`` and the swappable judge), and
reports results broken down by ``query_type`` — including abstention accuracy on
``unanswerable`` items.

Detachable by design: this imports FROM the pipeline; the pipeline never imports
from ``eval/``. Judge calls are cached on disk (a re-run costs zero API calls)
and rate-limited; any judge failure degrades gracefully to
``evaluation unavailable: <reason>`` rather than crashing or hanging.

    python -m eval.run --repo owner/name --dataset eval/datasets/<name>.jsonl \
        --metrics faithfulness,answer_relevancy,citation_precision,recall_at_k \
        --subset dev|full --out results/<run-id>/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import config
from core import manifest as manifest_mod
from core.context import RepositoryContext
from core.repo_url import parse_repo_url
from eval import metrics
from eval.golden_set import (GoldEntry, category_distribution, load_golden_set,
                             validate_golden_set)

DEV_SUBSET_SIZE = 10
ALL_METRICS = ["faithfulness", "answer_relevancy", "citation_precision",
               "citation_recall", "recall_at_k", "mrr", "ndcg"]

_ABSTAIN_RE = re.compile(
    r"\b(do(?:es)?\s+not\s+(?:contain|cover|mention|include)|insufficient|"
    r"cannot\s+(?:answer|provide|find|determine)|not\s+covered|outside\s+the\s+"
    r"indexed|no\s+(?:information|evidence)|declin\w+\s+to\s+answer|"
    r"unable\s+to\s+answer)\b", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def canonical_id(cid: str) -> str:
    """Canonicalise an id for comparison: strip split suffix; shorten commit sha."""
    base = str(cid).split("#", 1)[0]
    if base.startswith("commit_"):
        return "commit_" + base[len("commit_"):][:7]
    if base.isdigit():          # a bare PR/issue number -> assume PR
        return f"pr_{base}"
    return base


def looks_like_abstention(text: str) -> bool:
    return bool(_ABSTAIN_RE.search(text or ""))


def did_abstain(pr) -> bool:
    """Whether the system declined to give a confident, verified answer."""
    if getattr(pr, "empty", False) or getattr(pr, "refusal", False):
        return True
    if not getattr(pr, "guard_pass", False):
        return True
    return looks_like_abstention(getattr(pr, "text", ""))


class CachedJudge:
    """Disk-cached, rate-limited text->text judge wrapper."""

    def __init__(self, cache_path: Path, sleep_seconds: float):
        self.cache_path = cache_path
        self.sleep_seconds = sleep_seconds
        self._cache: dict[str, str] = {}
        if cache_path.exists():
            try:
                self._cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._cache = {}
        self._judge = None

    def __call__(self, prompt: str) -> str:
        key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if key in self._cache:
            return self._cache[key]
        if self._judge is None:
            self._judge = metrics.make_judge()
        raw = self._judge(prompt)          # may raise -> caller degrades
        self._cache[key] = raw
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache), encoding="utf-8")
        except OSError:
            pass
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)
        return raw


def _cited_ids(pr) -> list[str]:
    r = getattr(pr, "ref_report", None)
    if r is None:
        return []
    return list(r.valid_citations) + list(r.invalid_citations)


# --------------------------------------------------------------------------- #
# Core evaluation
# --------------------------------------------------------------------------- #
def metrics_for_entry(pr, entry: GoldEntry, metrics_list: list[str], judge,
                      judge_state: dict, latency_ms: float) -> dict:
    """Compute the metric row for one entry from an already-produced result.

    Shared by the evaluation runner and the ablation harness so both score
    entries identically.
    """
    row: dict = {"id": entry.id, "query_type": entry.query_type,
                 "latency_ms": latency_ms, "answer": pr.text,
                 "guard_pass": bool(getattr(pr, "guard_pass", False)),
                 "refusal": bool(getattr(pr, "refusal", False)),
                 # Per-row provenance. The configured model is NOT the model
                 # that answered: when a provider's daily quota is exhausted the
                 # pipeline falls back to local mid-run, and a report claiming
                 # the cloud model would be wrong for most of its own rows.
                 "model": getattr(pr, "model", ""),
                 "fell_back": bool(getattr(pr, "fell_back", False))}

    if entry.is_unanswerable:
        abstained = did_abstain(pr)
        row["abstained"] = abstained
        row["correct_abstention"] = abstained      # correct outcome == abstain
        return row

    ev = [canonical_id(x) for x in entry.evidence]
    ret = [canonical_id(c["chunk_id"]) for c in pr.chunks]
    if "recall_at_k" in metrics_list:
        row["recall_at_k"] = metrics.recall_at_k(ret, ev, config.FINAL_TOP_K)
    row["mrr"] = metrics.mrr(ret, ev)
    row["ndcg"] = metrics.ndcg_at_k(ret, ev, config.FINAL_TOP_K)
    cited = [canonical_id(c) for c in _cited_ids(pr)]
    cp, cr = metrics.citation_precision_recall(cited, ev)
    row["citation_precision"], row["citation_recall"] = cp, cr

    wants_judge = ("faithfulness" in metrics_list
                   or "answer_relevancy" in metrics_list)
    if wants_judge and judge_state.get("available", True):
        scored = metrics.judge_faithfulness_relevancy(
            entry.question, pr.text, [c["text"] for c in pr.chunks],
            judge_fn=judge)
        if scored["faithfulness"] is None and "error" in scored["judge"]:
            judge_state["available"] = False
            judge_state["reason"] = scored["judge"]
        row["faithfulness"] = scored["faithfulness"]
        row["answer_relevancy"] = scored["answer_relevancy"]
    return row


def evaluate_entry(pipeline_mod, retriever, answerer, nli, entry: GoldEntry,
                   since: str, until: str, metrics_list: list[str],
                   judge, judge_state: dict) -> dict:
    """Run the live pipeline for one entry and score it."""
    t0 = time.perf_counter()
    pr = pipeline_mod.answer_query(retriever, answerer, nli, entry.question,
                                   since, until)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return metrics_for_entry(pr, entry, metrics_list, judge, judge_state,
                             latency_ms)


def _mean(rows: list[dict], key: str):
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return statistics.fmean(vals) if vals else None


def _p95(rows: list[dict], key: str):
    vals = sorted(r[key] for r in rows if isinstance(r.get(key), (int, float)))
    if not vals:
        return None
    idx = min(len(vals) - 1, int(0.95 * (len(vals) - 1) + 0.999))
    return vals[idx]


def answered_by(rows: list[dict]) -> dict[str, int]:
    """Count how many answers each model actually produced.

    A long run routinely spans two models: the cloud provider answers until its
    daily quota runs out, then everything after that is local. Reporting only
    the configured model would misdescribe most of the run, so the real split is
    recorded and printed.
    """
    counts: dict[str, int] = {}
    for r in rows:
        model = r.get("model") or "(unrecorded)"
        counts[model] = counts.get(model, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def aggregate(rows: list[dict]) -> dict:
    by_type: dict[str, dict] = {}
    types = sorted({r["query_type"] for r in rows})
    for qt in types:
        group = [r for r in rows if r["query_type"] == qt]
        summary = {"n": len(group)}
        if qt == "unanswerable":
            correct = sum(1 for r in group if r.get("correct_abstention"))
            summary["abstention_accuracy"] = correct / len(group) if group else 0.0
            summary["hallucinated"] = len(group) - correct
        else:
            for k in ("recall_at_k", "mrr", "ndcg", "citation_precision",
                      "citation_recall", "faithfulness", "answer_relevancy"):
                summary[k] = _mean(group, k)
        summary["latency_ms_mean"] = _mean(group, "latency_ms")
        summary["latency_ms_p95"] = _p95(group, "latency_ms")
        by_type[qt] = summary
    return by_type


def format_report(repo: str, by_type: dict, judge_state: dict,
                  n_total: int, rows: list[dict] | None = None) -> str:
    lines = [f"Evaluation — {repo}   ({n_total} questions)",
             "=" * 60]
    if rows:
        counts = answered_by(rows)
        if counts:
            lines.append("\nanswers actually generated by:")
            for model, n in counts.items():
                lines.append(f"  {n:>4} × {model}")
            if len(counts) > 1:
                lines.append("  (a split means a provider quota ran out "
                             "mid-run and the pipeline fell back)")
    for qt, s in by_type.items():
        lines.append(f"\n[{qt}]  n={s['n']}")
        if qt == "unanswerable":
            lines.append(f"  abstention_accuracy : {s['abstention_accuracy']:.3f} "
                         f"({s['hallucinated']} hallucinated)")
        else:
            for k in ("recall_at_k", "mrr", "ndcg", "citation_precision",
                      "citation_recall", "faithfulness", "answer_relevancy"):
                v = s.get(k)
                lines.append(f"  {k:<20}: "
                             + ("n/a" if v is None else f"{v:.3f}"))
        lm, lp = s.get("latency_ms_mean"), s.get("latency_ms_p95")
        lines.append(f"  latency_ms mean/p95 : "
                     + (f"{lm:.0f} / {lp:.0f}" if lm is not None else "n/a"))
    if not judge_state.get("available", True):
        lines.append(f"\nevaluation unavailable (judge): {judge_state.get('reason')}")
    return "\n".join(lines)


def run(repo: str, dataset: str, metrics_list: list[str], subset: str,
        out_dir: Path, sleep: float, limit: int | None) -> int:
    import query_pipeline  # imported here: eval depends on pipeline, not vice-versa

    ref = parse_repo_url(repo)
    ctx = RepositoryContext.for_ref(ref)
    if not ctx.manifest_path.exists():
        print(f"{ref.full_name} is not indexed. Index it first "
              f"(python -m jobs.runner {repo}).", file=sys.stderr)
        return 1
    m = manifest_mod.read_manifest(ctx.manifest_path)
    reusable, reason = manifest_mod.is_reusable(m)
    if not reusable:
        print(f"Index for {ref.full_name} is not usable: {reason}", file=sys.stderr)
        return 1
    since = m["coverage"].get("since", "")
    until = m["coverage"].get("until", "")

    entries = load_golden_set(dataset)
    validate_golden_set(entries)             # fails loudly before any query
    if subset == "dev":
        entries = entries[:DEV_SUBSET_SIZE]
    if limit:
        entries = entries[:limit]
    if not entries:
        print("No entries to evaluate.", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    from generation.answerer import Answerer
    from guard.nli_verifier import NLIVerifier
    from retrieval.retriever import Retriever

    retriever, answerer, nli = Retriever(ctx), Answerer(), NLIVerifier()
    judge = CachedJudge(out_dir / "judge_cache.json", sleep)
    judge_state: dict = {"available": True}

    # Resume support: completed rows are checkpointed after every question, so a
    # long run that is interrupted (OOM kill, closed laptop, Ctrl-C) can be
    # re-invoked and will skip the questions it already scored.
    partial_path = out_dir / "rows_partial.jsonl"
    rows: list[dict] = []
    done_ids: set[str] = set()
    if partial_path.exists():
        for line in partial_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue          # ignore a torn final line from a hard kill
            rows.append(row)
            done_ids.add(row.get("id"))
        if done_ids:
            print(f"  resuming: {len(done_ids)} question(s) already scored",
                  file=sys.stderr)

    for i, e in enumerate(entries, 1):
        if e.id in done_ids:
            continue
        print(f"  [{i}/{len(entries)}] {e.query_type}: {e.question[:60]}",
              file=sys.stderr)
        row = evaluate_entry(query_pipeline, retriever, answerer, nli, e,
                             since, until, metrics_list, judge, judge_state)
        rows.append(row)
        with partial_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_type = aggregate(rows)
    report = format_report(ref.full_name, by_type, judge_state, len(rows), rows)
    print(report)

    (out_dir / "results.json").write_text(json.dumps({
        "repo": ref.full_name, "dataset": str(dataset),
        "distribution": category_distribution(entries),
        "by_query_type": by_type, "rows": rows,
        "judge": judge_state, "created_at": datetime.now(timezone.utc).isoformat(),
        # Provenance: scores are only comparable across runs that used the same
        # models, so record exactly which ones produced this run.
        # What ACTUALLY answered, counted per model. "generation" below is only
        # what was configured; on a long run the two routinely differ.
        "answered_by": answered_by(rows),
        "models": {
            # NOTE: this is what was CONFIGURED. For what actually answered —
            # which differs whenever a provider quota runs out mid-run — see
            # the "answered_by" counts above.
            "generation": config.effective_generation_model(),
            "embedding": config.EMBEDDING_MODEL,
            "reranker": config.RERANKER_MODEL,
            "nli": config.NLI_MODEL,
            "judge_provider": config.JUDGE_PROVIDER,
            "judge_model": config.judge_model_name(),
        },
        # Self-preference audit: the answerer, judge and question author must be
        # three different models, or faithfulness scores are inflated by a model
        # grading its own output / its own phrasing.
        "evaluation_roles": config.evaluation_roles(),
        "roles_distinct": dict(zip(("ok", "detail"), config.roles_are_distinct())),
    }, indent=2), encoding="utf-8")
    (out_dir / "report.txt").write_text(report, encoding="utf-8")
    partial_path.unlink(missing_ok=True)   # run completed; drop the checkpoint
    print(f"\nWrote {out_dir/'results.json'} and {out_dir/'report.txt'}",
          file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="RepoMind evaluation runner")
    p.add_argument("--repo", required=True, help="owner/name (must be indexed)")
    p.add_argument("--dataset", required=True, help="path to a golden-set JSONL")
    p.add_argument("--metrics", default=",".join(ALL_METRICS),
                   help="comma-separated metric names")
    p.add_argument("--subset", choices=["dev", "full"], default="full")
    p.add_argument("--out", default="", help="output directory")
    p.add_argument("--sleep", type=float, default=0.0,
                   help="seconds to sleep between judge calls")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    metrics_list = [x.strip() for x in args.metrics.split(",") if x.strip()]
    out_dir = Path(args.out) if args.out else (
        Path("results") / datetime.now().strftime("run-%Y%m%d-%H%M%S"))
    return run(args.repo, args.dataset, metrics_list, args.subset, out_dir,
               args.sleep, args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
