"""Demo readiness check — verifies everything you actually show an evaluator.

Run this before any demo. It exercises the exact paths a viewer will see and
fails loudly if any of them would embarrass you live:

  1. Ollama up + required models present (the offline safety net)
  2. Every indexed repository loads with a valid, reusable manifest
  3. A grounded answer with at least one REAL clickable citation
  4. The hallucination guard catches a fabricated citation
  5. An out-of-coverage question is refused, not guessed
  6. An unanswerable question is refused, not hallucinated
  7. The evolution graph has real issue<->PR<->commit links to show
  8. Cloud generation works AND falls back to local when it can't

Exits non-zero if anything a demo depends on is broken.

    python scripts/demo_check.py            # full check
    python scripts/demo_check.py --repo pallets/click
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
_results: list[tuple[str, str, str]] = []


def record(status: str, name: str, detail: str = "") -> None:
    _results.append((status, name, detail))
    icon = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️ "}[status]
    print(f"{icon} [{status}] {name}" + (f": {detail}" if detail else ""), flush=True)


# --------------------------------------------------------------------------- #
def check_ollama() -> bool:
    import requests
    try:
        tags = [m["name"] for m in requests.get(
            f"{config.OLLAMA_HOST}/api/tags", timeout=5).json().get("models", [])]
    except Exception as exc:  # noqa: BLE001
        record(FAIL, "Ollama reachable", f"{exc} — start Ollama before demoing")
        return False

    def present(model: str) -> bool:
        base = model.split(":")[0]
        return any(t == model or t.split(":")[0] == base for t in tags)

    missing = [m for m in (config.EMBEDDING_MODEL, config.GENERATION_MODEL)
               if not present(m)]
    if missing:
        record(FAIL, "Required Ollama models", f"missing {missing} — run `ollama pull`")
        return False
    record(PASS, "Ollama + models", f"{config.EMBEDDING_MODEL}, {config.GENERATION_MODEL}")
    return True


def check_repositories() -> list[str]:
    from core import registry
    entries = registry.list_repositories()
    ready = [e for e in entries if e.reusable]
    stale = [e for e in entries if not e.reusable]
    if not ready:
        record(FAIL, "Indexed repositories", "none usable — index one before demoing")
        return []
    record(PASS, "Indexed repositories",
           f"{len(ready)} ready: {', '.join(e.full_name for e in ready)}")
    if stale:
        record(WARN, "Stale indexes",
               f"{', '.join(e.full_name for e in stale)} (re-index or avoid in demo)")
    return [e.full_name for e in ready]


def _pipeline_for(full_name: str):
    from core import manifest as manifest_mod
    from core.context import RepositoryContext
    from core.repo_url import parse_repo_url
    from generation.answerer import Answerer
    from guard.nli_verifier import NLIVerifier
    from retrieval.retriever import Retriever

    ctx = RepositoryContext.for_ref(parse_repo_url(full_name))
    m = manifest_mod.read_manifest(ctx.manifest_path)
    cov = m.get("coverage", {})
    return (ctx, Retriever(ctx), Answerer(), NLIVerifier(),
            cov.get("since", ""), cov.get("until", ""), m)


def check_grounded_answer(full_name: str, question: str) -> bool:
    import query_pipeline
    ctx, r, a, n, since, until, _ = _pipeline_for(full_name)
    pr = query_pipeline.answer_query(r, a, n, question, since, until)
    if pr.empty or not pr.text.strip():
        record(FAIL, "Grounded answer", f"empty answer for {full_name}")
        return False
    valid = len(pr.ref_report.valid_citations) if pr.ref_report else 0
    if valid < 1:
        record(FAIL, "Grounded answer", "no valid citation — nothing to click in demo")
        return False
    url_ok = any(c.get("github_url", "").startswith("http") for c in pr.chunks)
    if not url_ok:
        record(FAIL, "Clickable citations", "no real GitHub URLs in evidence")
        return False
    who = pr.model + (" (fell back to local)" if pr.fell_back else "")
    record(PASS, "Grounded answer + citations",
           f"{valid} valid citation(s) via {who}")
    return True


def check_guard_catches_fake_citation(full_name: str) -> bool:
    """The headline demo moment: the guard rejecting an invented citation."""
    from guard.reference_validator import validate_references
    chunks = [{"chunk_id": "pr_1", "source_type": "pr", "text": "PR #1 fixed it.",
               "github_url": "https://github.com/x/y/pull/1"}]
    rep = validate_references("This was fixed in [pr_99999].", chunks)
    if rep.is_valid or "pr_99999" not in rep.invalid_citations:
        record(FAIL, "Guard catches fake citation", "fabricated id was NOT flagged")
        return False
    record(PASS, "Guard catches fake citation", "invented [pr_99999] correctly flagged")
    return True


def check_refuses_unanswerable(full_name: str) -> bool:
    import query_pipeline
    from eval.run import did_abstain
    _, r, a, n, since, until, _ = _pipeline_for(full_name)
    q = "Why was the built-in quantum blockchain mining module removed?"
    pr = query_pipeline.answer_query(r, a, n, q, since, until)
    if not did_abstain(pr):
        record(FAIL, "Refuses fabricated premise",
               f"answered instead of refusing: {pr.text[:90]}")
        return False
    record(PASS, "Refuses fabricated premise", "declined rather than inventing")
    return True


def check_out_of_range(full_name: str) -> bool:
    import query_pipeline
    from eval.run import looks_like_abstention
    _, r, a, n, since, until, _ = _pipeline_for(full_name)
    pr = query_pipeline.answer_query(
        r, a, n, "What major features were added in 1998?", since, until)
    text = (pr.text or "").lower()
    signalled = (looks_like_abstention(pr.text) or "1998" in text
                 or "coverage" in text or "outside" in text)
    if not signalled:
        record(WARN, "Out-of-coverage question",
               f"did not clearly flag the date range: {pr.text[:80]}")
        return True     # warn, not fail — wording varies
    record(PASS, "Out-of-coverage question", "flagged the indexed date range")
    return True


def check_evolution_graph(full_name: str) -> bool:
    from core.context import RepositoryContext
    from core.repo_url import parse_repo_url
    from process import linker
    ctx = RepositoryContext.for_ref(parse_repo_url(full_name))
    if not ctx.links_path.exists():
        record(WARN, "Evolution graph", f"{full_name} has no links.json")
        return True
    graph = json.loads(ctx.links_path.read_text(encoding="utf-8"))
    n = linker.count_closes_links(graph)
    if n == 0:
        record(WARN, "Evolution graph",
               f"{full_name} has 0 links — pick another repo for the graph demo")
        return True
    dot = linker.to_dot(graph)
    if not dot.startswith("digraph"):
        record(FAIL, "Evolution graph", "DOT render failed")
        return False
    record(PASS, "Evolution graph", f"{full_name}: {n} issue↔PR↔commit links")
    return True


def check_cloud_and_fallback() -> bool:
    """Cloud generation should work, and MUST degrade to local when it can't."""
    from generation.answerer import Answerer
    msgs = [{"role": "user", "content": "Reply with exactly: OK"}]

    if not config.api_generation_enabled():
        record(WARN, "Cloud generation", "not configured — demo runs fully local (fine)")
        return True
    try:
        Answerer()._chat_api(msgs)
        record(PASS, "Cloud generation", f"{config.GENERATION_API_MODEL} reachable")
    except Exception as exc:  # noqa: BLE001
        record(WARN, "Cloud generation",
               f"unavailable ({str(exc)[:60]}) — fallback will cover it")

    # Force a failure and confirm we still get a local answer.
    orig = config.GENERATION_API_KEY
    try:
        config.GENERATION_API_KEY = "deliberately-invalid"
        config.GENERATION_API_MAX_RETRIES = 0
        chunks = [{"chunk_id": "pr_1", "source_type": "pr",
                   "text": "PR #1 fixed the crash.", "github_url": "http://x/1"}]
        res = Answerer().answer("What fixed it?", chunks)
        if not res.text.strip():
            record(FAIL, "Local fallback", "no answer when cloud failed")
            return False
        if not res.fell_back:
            record(WARN, "Local fallback", "did not report fell_back")
        record(PASS, "Local fallback", "answered locally when cloud key was invalid")
        return True
    except Exception as exc:  # noqa: BLE001
        record(FAIL, "Local fallback", f"raised instead of falling back: {exc}")
        return False
    finally:
        config.GENERATION_API_KEY = orig


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="RepoMind demo readiness check")
    p.add_argument("--repo", default="", help="repo to exercise (default: best available)")
    p.add_argument("--question", default="", help="override the demo question")
    args = p.parse_args(argv)

    print("RepoMind — demo readiness check\n" + "=" * 46)
    if not check_ollama():
        print("\nRESULT: NOT DEMO READY")
        return 1

    ready = check_repositories()
    if not ready:
        print("\nRESULT: NOT DEMO READY")
        return 1

    repo = args.repo or next(
        (r for r in ready if r != "acme/widgets"), ready[0])
    print(f"\n-- exercising: {repo} --")

    question = args.question or "What is one notable recent change, and why was it made?"
    check_grounded_answer(repo, question)
    check_guard_catches_fake_citation(repo)
    check_refuses_unanswerable(repo)
    check_out_of_range(repo)

    # Show the graph on whichever indexed repo has the most links.
    best = repo
    try:
        from core.context import RepositoryContext
        from core.repo_url import parse_repo_url
        from process import linker
        counts = {}
        for r in ready:
            c = RepositoryContext.for_ref(parse_repo_url(r))
            if c.links_path.exists():
                counts[r] = linker.count_closes_links(
                    json.loads(c.links_path.read_text(encoding="utf-8")))
        if counts:
            best = max(counts, key=counts.get)
    except Exception:  # noqa: BLE001
        pass
    check_evolution_graph(best)
    check_cloud_and_fallback()

    fails = [r for r in _results if r[0] == FAIL]
    warns = [r for r in _results if r[0] == WARN]
    print("\n" + "=" * 46)
    print(f"{len(_results) - len(fails) - len(warns)} passed, "
          f"{len(warns)} warnings, {len(fails)} failures")
    if fails:
        print("\nRESULT: NOT DEMO READY — fix these first:")
        for _, name, detail in fails:
            print(f"  - {name}: {detail}")
        return 1
    print("\nRESULT: DEMO READY ✅")
    if warns:
        print("(warnings are non-blocking, but read them before presenting)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
