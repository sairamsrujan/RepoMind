"""Auto-generate a repository-dependent golden set (standalone offline tool).

Synthesizes N questions for one indexed repository, grounded ONLY in that
repository's real indexed chunks (commits, issues, releases, and PRs/reviews
where present) — never generic or templated across repos. Every entry keeps
the existing golden-set taxonomy (see eval/golden_set.py):
factual / causal / cross_commit / evolution / unanswerable, split according to
the target distribution documented in the README (25/15/25/20/15%).

Method, per category (all candidate selection is plain Python/regex over real
chunks — only the final question/answer wording is LLM-synthesized, and always
grounded in the real evidence text supplied in the prompt):

  factual      - one substantive commit/issue/release chunk
  causal       - chunks whose text carries rationale language (fix/because/
                 deprecate/since/resolve) or an explicit issue reference
  cross_commit - real linked clusters (issue + the PR/commits that closed it,
                 from the link graph) where available, else a release's own
                 multi-item changelog, else commits sharing a PR number
  evolution    - chunks that share a distinctive keyword across >=2 dates,
                 i.e. a real feature/fix touched more than once over time
  unanswerable - plausible-sounding but fictional premises for THIS repo's
                 domain, each rejected unless a real retrieval sanity check
                 confirms nothing closely matches (so it's a genuine abstention
                 test, not just an untested guess)

This is a standalone tool (like eval/ablation.py): it imports FROM the live
pipeline (retrieval/generation) for the unanswerable sanity check and for
calling the local Ollama model, but nothing in the pipeline imports this file.

Usage:
    python -m eval.generate_golden_set --repo owner/name --n 50 \
        --out eval/datasets/<slug>.jsonl [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests

import config
from core.context import RepositoryContext
from core.repo_url import parse_repo_url
from eval.golden_set import GoldEntry, validate_golden_set
from process import chunker

CATEGORY_WEIGHTS = {
    "factual": 0.25, "causal": 0.15, "cross_commit": 0.25,
    "evolution": 0.20, "unanswerable": 0.15,
}

_NOISE_TITLE_RE = re.compile(
    r"^(?:[^\w]*)(update release notes|bump version|release version|"
    r"merge (branch|pull request)|update translations?)\b", re.IGNORECASE)
_RATIONALE_RE = re.compile(
    r"\b(fix(?:e[sd])?|clos(?:e[sd])?|resolv\w*|because|deprecat\w*|since|refs?)\b",
    re.IGNORECASE)
_PR_NUM_RE = re.compile(r"#(\d+)")
_STOPWORDS = {
    "the", "a", "an", "to", "of", "for", "and", "in", "on", "with", "is",
    "are", "this", "that", "by", "update", "updates", "release", "notes",
    "from", "into", "when", "not", "now", "new", "via", "was", "were",
}


# --------------------------------------------------------------------------- #
# Category target counts (largest-remainder rounding so they sum exactly to n)
# --------------------------------------------------------------------------- #
def category_targets(n: int) -> dict[str, int]:
    raw = {k: n * w for k, w in CATEGORY_WEIGHTS.items()}
    floors = {k: int(v) for k, v in raw.items()}
    remainder = n - sum(floors.values())
    order = sorted(raw, key=lambda k: raw[k] - floors[k], reverse=True)
    for k in order[:remainder]:
        floors[k] += 1
    return floors


# --------------------------------------------------------------------------- #
# Candidate selection (pure Python — no LLM calls here)
# --------------------------------------------------------------------------- #
def _is_substantive(c: dict) -> bool:
    if c["source_type"] not in ("commit", "issue", "release", "pr", "review"):
        return False
    if _NOISE_TITLE_RE.search(c.get("title", "") or ""):
        return False
    return int(c.get("token_count", 0)) >= 6


def factual_candidates(chunks: list[dict]) -> list[list[dict]]:
    return [[c] for c in chunks if _is_substantive(c)]


def causal_candidates(chunks: list[dict]) -> list[list[dict]]:
    return [[c] for c in chunks
            if c["source_type"] in ("commit", "issue", "pr") and _is_substantive(c)
            and _RATIONALE_RE.search(c.get("text", ""))]


def _base_id(cid: str) -> str:
    return cid.split("#", 1)[0]


def closes_link_clusters(graph: dict, chunk_by_base: dict[str, list[dict]]
                         ) -> list[list[dict]]:
    clusters = []
    for num, info in graph.get("issues", {}).items():
        ids = [f"issue_{num}"]
        ids += [f"pr_{p}" for p in info.get("closed_by_prs", [])]
        ids += [f"commit_{sha[:12]}" for sha in info.get("closed_by_commits", [])]
        group = [c for cid in ids for c in chunk_by_base.get(cid, [])]
        if len(group) >= 2:  # a real link, not a lone issue
            clusters.append(group)
    return clusters


def release_group_clusters(chunks: list[dict]) -> list[list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        if c["source_type"] == "release":
            groups[_base_id(c["chunk_id"])].append(c)
    return [g for g in groups.values() if sum(x["token_count"] for x in g) > 40]


def pr_number_clusters(chunks: list[dict]) -> list[list[dict]]:
    by_pr: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        if c["source_type"] != "commit":
            continue
        for m in _PR_NUM_RE.finditer(c.get("title", "") or ""):
            by_pr[m.group(1)].append(c)
    return [g for g in by_pr.values() if len(g) >= 2]


def cross_commit_candidates(chunks: list[dict], graph: dict | None) -> list[list[dict]]:
    chunk_by_base: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        chunk_by_base[_base_id(c["chunk_id"])].append(c)
    clusters = []
    if graph:
        clusters += closes_link_clusters(graph, chunk_by_base)
    clusters += release_group_clusters(chunks)
    clusters += pr_number_clusters(chunks)
    return clusters


def evolution_candidates(chunks: list[dict]) -> list[list[dict]]:
    keyed: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        if c["source_type"] not in ("commit", "issue", "release"):
            continue
        words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{3,}", (c.get("title") or "").lower())
        for w in {w for w in words if w not in _STOPWORDS}:
            keyed[w].append(c)
    clusters, seen = [], set()
    for group in keyed.values():
        dates = {g["date"][:10] for g in group if g.get("date")}
        if len(group) >= 2 and len(dates) >= 2:
            ids = tuple(sorted(g["chunk_id"] for g in group))
            if ids not in seen:
                seen.add(ids)
                clusters.append(sorted(group, key=lambda x: x.get("date", "")))
    return clusters


# --------------------------------------------------------------------------- #
# LLM synthesis (grounded strictly in the supplied real evidence)
# --------------------------------------------------------------------------- #
_GEN_PROMPT = """You are creating an evaluation question for a QA system about \
the GitHub repository {repo}.

Below is REAL evidence from the repository's history. Based ONLY on this \
evidence, write:
1. One natural, specific question a developer might ask about why/how/what \
happened (a {qtype}-style question).
2. A one-to-two sentence ground-truth answer, using ONLY the evidence below. \
Do not invent anything not stated in the evidence.

Return ONLY a JSON object: {{"question": "...", "ground_truth": "..."}}

EVIDENCE:
{evidence}
"""

_UNANSWERABLE_PROMPT = """The repository "{repo}" is described by this real \
sample of its recent history:
{sample}

Invent {n} short, plausible-sounding but FICTIONAL feature or design-decision \
names that would NOT actually exist in a project like this one. For each, \
phrase it as a question like "Why was X added/removed/changed?". Return ONLY \
a JSON array of {n} question strings, nothing else.
"""


def _extract_json(text: str):
    m = re.search(r"[\{\[].*[\}\]]", text or "", re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _ollama_chat(prompt: str, model: str, temperature: float = 0.3) -> str:
    """Chat completion for question synthesis.

    Routed through ``QUESTIONGEN_PROVIDER``, which is deliberately a *different*
    provider from ``JUDGE_PROVIDER``: if one model both writes the questions and
    grades the answers it favours its own phrasing (self-preference bias), which
    inflates the reported scores. Falls back to local on any failure.
    """
    import providers

    # Pin the exact model: question-gen and the judge may share a provider, and
    # letting both fall through to that provider's default would silently make
    # the question author and the grader the same model again.
    text, _used = providers.chat(config.QUESTIONGEN_PROVIDER, prompt,
                                 model=config.questiongen_model_name() or None,
                                 temperature=temperature)
    return text


def synthesize_entry(cluster: list[dict], qtype: str, repo: str, model: str,
                     idx: int) -> GoldEntry | None:
    evidence_text = "\n---\n".join(c["text"][:1200] for c in cluster)[:5000]
    raw = _ollama_chat(
        _GEN_PROMPT.format(repo=repo, qtype=qtype, evidence=evidence_text), model)
    parsed = _extract_json(raw)
    if not parsed or not parsed.get("question") or not parsed.get("ground_truth"):
        return None
    return GoldEntry(
        id=f"auto-{qtype}-{idx:03d}", question=parsed["question"].strip(),
        query_type=qtype, ground_truth=parsed["ground_truth"].strip(),
        evidence=[c["chunk_id"] for c in cluster],
        notes=f"auto-generated from {len(cluster)} real chunk(s): "
              f"{', '.join(c['chunk_id'] for c in cluster[:4])}",
    )


# Ask for questions in small batches. A single call asking for N questions is
# capped by GENERATION_MAX_TOKENS, so large targets silently come back short (or
# as truncated, unparseable JSON). Batching also varies the evidence sample per
# round, which yields more varied fictional premises than one big request.
_UNANSWERABLE_BATCH = 8
_UNANSWERABLE_MAX_ROUNDS = 12


def generate_unanswerable(repo: str, chunks: list[dict], target: int, model: str,
                          seed: int, log=lambda msg: None) -> list[GoldEntry]:
    """Invent fictional premises and keep only those retrieval confirms absent.

    Each candidate is searched for in the real index; anything that matches
    real content is discarded, so every surviving entry is a genuine abstention
    test rather than an untested guess.
    """
    from retrieval.retriever import Retriever

    sample_pool = [c for c in chunks if _is_substantive(c)]
    ctx = RepositoryContext.for_ref(parse_repo_url(repo))
    retriever = Retriever(ctx)

    entries: list[GoldEntry] = []
    seen: set[str] = set()
    rejected = 0

    for round_no in range(_UNANSWERABLE_MAX_ROUNDS):
        if len(entries) >= target:
            break
        # Re-sample the evidence each round so the model sees a different slice.
        rnd = random.Random(seed + round_no)
        sample = "\n".join(f"- {c['title']}" for c in rnd.sample(
            sample_pool, min(15, len(sample_pool)))) if sample_pool else "(no data)"

        raw = _ollama_chat(
            _UNANSWERABLE_PROMPT.format(repo=repo, sample=sample,
                                        n=_UNANSWERABLE_BATCH), model,
            temperature=0.8)   # higher temp -> more varied inventions
        candidates = _extract_json(raw) or []
        candidates = [q for q in candidates if isinstance(q, str) and q.strip()]

        for q in candidates:
            if len(entries) >= target:
                break
            key = q.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            hits = retriever.retrieve(q, top_k=3)
            top_score = max((h.get("rerank_score", 0.0) for h in hits), default=0.0)
            if top_score >= 0.5:   # suspiciously relevant -> not truly absent
                rejected += 1
                continue
            entries.append(GoldEntry(
                id=f"auto-unanswerable-{len(entries):03d}", question=q.strip(),
                query_type="unanswerable", ground_truth=None, evidence=[],
                notes=f"LLM-invented fictional premise, verified absent via "
                      f"retrieval (top rerank score {top_score:.2f})",
            ))
        log(f"  unanswerable round {round_no + 1}: {len(entries)}/{target} kept, "
            f"{rejected} rejected as too-relevant")
    return entries


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def generate_abstention_set(repo: str, n: int, seed: int = 42,
                            model: str | None = None,
                            log=lambda msg: None) -> list[GoldEntry]:
    """Build a dedicated, larger set of verified-unanswerable questions.

    Abstention accuracy is the headline claim, and a handful of questions per
    repository is a thin basis for it. Unanswerable entries skip the LLM judge
    entirely (see ``eval/run.metrics_for_entry``), so a large abstention set
    costs local pipeline time but **no API quota** — which makes it the cheapest
    way to put the headline number on a defensible sample size.
    """
    model = model or config.GENERATION_MODEL
    ctx = RepositoryContext.for_ref(parse_repo_url(repo))
    chunks = chunker.load_chunks(ctx)
    if not chunks:
        raise RuntimeError(f"{repo} has no indexed chunks — index it first.")
    return generate_unanswerable(repo, chunks, n, model, seed, log=log)


def generate_golden_set(repo: str, n: int, seed: int = 42,
                        model: str | None = None,
                        log=lambda msg: None) -> list[GoldEntry]:
    model = model or config.GENERATION_MODEL
    ref = parse_repo_url(repo)
    ctx = RepositoryContext.for_ref(ref)
    chunks = chunker.load_chunks(ctx)
    if not chunks:
        raise RuntimeError(f"{repo} has no indexed chunks — index it first.")
    graph = (json.loads(ctx.links_path.read_text(encoding="utf-8"))
            if ctx.links_path.exists() else None)

    rnd = random.Random(seed)
    targets = category_targets(n)
    results: list[GoldEntry] = []

    for qtype, target in targets.items():
        if target == 0:
            continue
        if qtype == "unanswerable":
            got = generate_unanswerable(repo, chunks, target, model, seed, log=log)
            results += got
            log(f"[{qtype}] {len(got)}/{target}")
            continue

        if qtype == "factual":
            cands = factual_candidates(chunks)
        elif qtype == "causal":
            cands = causal_candidates(chunks)
        elif qtype == "cross_commit":
            cands = cross_commit_candidates(chunks, graph)
        else:  # evolution
            cands = evolution_candidates(chunks)

        rnd.shuffle(cands)
        produced, idx = 0, 0
        for cluster in cands:
            if produced >= target:
                break
            idx += 1
            entry = synthesize_entry(cluster, qtype, repo, model, idx)
            if entry:
                results.append(entry)
                produced += 1
        log(f"[{qtype}] {produced}/{target}"
            + ("" if produced >= target else
               f"  (real-data limit: only {len(cands)} candidate cluster(s) "
               f"available)"))

    return results


def save_golden_set(entries: list[GoldEntry], out_path: Path) -> None:
    validate_golden_set(entries)  # fail loudly before writing anything bad
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Auto-generate a golden set")
    p.add_argument("--repo", required=True)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--out", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--only-unanswerable", action="store_true",
                   help="generate ONLY verified-unanswerable questions, for a "
                        "dedicated abstention benchmark (costs no judge quota)")
    args = p.parse_args(argv)

    ref = parse_repo_url(args.repo)
    if args.out:
        out_path = Path(args.out)
    elif args.only_unanswerable:
        out_path = Path("eval") / "datasets" / f"{ref.slug}_abstention.jsonl"
    else:
        out_path = Path("eval") / "datasets" / f"{ref.slug}.jsonl"

    kind = "unanswerable" if args.only_unanswerable else "mixed"
    print(f"Generating {args.n} {kind} questions for {ref.full_name} "
          f"-> {out_path}", file=sys.stderr)
    gen = generate_abstention_set if args.only_unanswerable else generate_golden_set
    entries = gen(args.repo, args.n, seed=args.seed,
                  log=lambda msg: print(f"  {msg}", file=sys.stderr))
    save_golden_set(entries, out_path)
    print(f"Wrote {len(entries)}/{args.n} entries to {out_path}", file=sys.stderr)
    return 0 if len(entries) == args.n else 1


if __name__ == "__main__":
    raise SystemExit(main())
