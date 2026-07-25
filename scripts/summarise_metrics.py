"""Summarise the Phase A per-query metrics log.

Reads ``data/metrics/queries.jsonl`` and prints a plain-text table: count,
mean, median, and p95 for each latency, plus the guard pass rate and citation
validity rate. Standard library only (no pandas).

Usage:
    python scripts/summarise_metrics.py [path/to/queries.jsonl]
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

# Make the project root importable so we can reuse the configured metrics path.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402

LATENCY_FIELDS = ["retrieval_ms", "rerank_ms", "generation_ms", "guard_ms",
                  "total_ms"]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = pct / 100.0 * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def load_records(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # skip a malformed line rather than abort
    return records


def summarise(records: list[dict]) -> str:
    if not records:
        return "No metrics recorded yet."

    answered = [r for r in records if not r.get("empty")]
    lines = [f"RepoMind query metrics — {len(records)} queries "
             f"({len(answered)} answered, {len(records) - len(answered)} empty)",
             "=" * 66]

    # Latency table.
    header = f"{'metric':<16}{'count':>7}{'mean':>10}{'median':>10}{'p95':>10}"
    lines += [header, "-" * len(header)]
    for field in LATENCY_FIELDS:
        vals = [float(r[field]) for r in records if isinstance(r.get(field), (int, float))]
        if not vals:
            lines.append(f"{field:<16}{0:>7}{'-':>10}{'-':>10}{'-':>10}")
            continue
        lines.append(
            f"{field:<16}{len(vals):>7}{statistics.fmean(vals):>10.1f}"
            f"{statistics.median(vals):>10.1f}{_percentile(vals, 95):>10.1f}"
        )

    # Guard + citation rates (over answered queries).
    lines += ["-" * len(header)]
    if answered:
        passes = sum(1 for r in answered if r.get("guard_verdict") == "pass")
        total_cites = sum(int(r.get("num_citations", 0)) for r in answered)
        valid_cites = sum(int(r.get("num_valid_citations", 0)) for r in answered)
        lines.append(f"guard pass rate      : {passes}/{len(answered)} "
                     f"({100.0 * passes / len(answered):.1f}%)")
        if total_cites:
            lines.append(f"citation validity    : {valid_cites}/{total_cites} "
                         f"({100.0 * valid_cites / total_cites:.1f}%)")
        else:
            lines.append("citation validity    : (no citations recorded)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    path = Path(argv[0]) if argv else config.METRICS_PATH
    print(summarise(load_records(path)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
