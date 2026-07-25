"""Phase G2: one-command liveness check for the whole system.

Run this once a month during the gap before the demo. It verifies:
  1. Ollama is reachable.
  2. Both required Ollama model tags are present.
  3. The reranker and NLI checkpoints resolve from the local HuggingFace cache
     WITHOUT network access (fails loudly if they would need to download).
  4. `pytest -q` passes.
  5. One end-to-end query against a pre-indexed repository returns a non-empty
     answer with at least one valid citation.

Prints PASS/FAIL per check and exits non-zero on any failure.

    python scripts/smoke_test.py [--no-tests]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402


def _ollama_tags() -> list[str]:
    import requests
    r = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=5)
    r.raise_for_status()
    return [m.get("name", "") for m in r.json().get("models", [])]


def check_ollama_reachable() -> tuple[bool, str]:
    try:
        _ollama_tags()
        return True, f"Ollama reachable at {config.OLLAMA_HOST}"
    except Exception as exc:  # noqa: BLE001
        return False, f"Ollama not reachable: {exc}"


def check_models_present() -> tuple[bool, str]:
    try:
        tags = _ollama_tags()
    except Exception as exc:  # noqa: BLE001
        return False, f"could not list models: {exc}"

    def present(model: str) -> bool:
        # Ollama tags include an implicit ":latest"; match on the base too.
        base = model.split(":")[0]
        return any(t == model or t.split(":")[0] == base for t in tags)

    missing = [m for m in (config.EMBEDDING_MODEL, config.GENERATION_MODEL)
               if not present(m)]
    if missing:
        return False, f"missing Ollama models: {missing} (run `ollama pull`)"
    return True, f"models present: {config.EMBEDDING_MODEL}, {config.GENERATION_MODEL}"


def check_hf_models_cached() -> tuple[bool, str]:
    # Force offline: loading must succeed purely from the local cache.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        from sentence_transformers import CrossEncoder
        CrossEncoder(config.RERANKER_MODEL, max_length=512)
        CrossEncoder(config.NLI_MODEL, max_length=512)
        return True, "reranker + NLI resolved from local cache (offline)"
    except Exception as exc:  # noqa: BLE001
        return False, (f"reranker/NLI NOT cached — would need to download "
                       f"({config.RERANKER_MODEL} / {config.NLI_MODEL}): {exc}")


def check_pytest() -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        cwd=str(_ROOT), capture_output=True, text=True)
    last = (proc.stdout.strip().splitlines() or ["(no output)"])[-1]
    return proc.returncode == 0, f"pytest: {last}"


def check_end_to_end() -> tuple[bool, str]:
    from core import registry
    from core.context import RepositoryContext
    from core.repo_url import parse_repo_url

    # Prefer the deterministic demo repo; else the first ready+reusable repo.
    ctx = None
    question = "What is one notable recent change, and why was it made?"
    demo = RepositoryContext.for_ref(parse_repo_url("acme/widgets"))
    if demo.manifest_path.exists():
        ctx, question = demo, "What fixed the startup crash?"
    else:
        for e in registry.list_repositories():
            if e.reusable:
                ctx = RepositoryContext.for_ref(parse_repo_url(e.full_name))
                break
    if ctx is None:
        return False, "no indexed repository to query (index one first)"

    from core import manifest as manifest_mod
    from generation.answerer import Answerer
    from guard.nli_verifier import NLIVerifier
    from retrieval.retriever import Retriever
    import query_pipeline

    m = manifest_mod.read_manifest(ctx.manifest_path)
    since, until = m["coverage"].get("since", ""), m["coverage"].get("until", "")
    pr = query_pipeline.answer_query(Retriever(ctx), Answerer(), NLIVerifier(),
                                     question, since, until)
    if pr.empty or not pr.text.strip():
        return False, f"empty answer for {ctx.slug!r}"
    valid = len(pr.ref_report.valid_citations) if pr.ref_report else 0
    if valid < 1:
        return False, f"answer had no valid citation ({ctx.slug})"
    return True, f"end-to-end OK on {ctx.slug} ({valid} valid citation(s))"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="RepoMind smoke test")
    p.add_argument("--no-tests", action="store_true",
                   help="skip the pytest check (faster)")
    args = p.parse_args(argv)

    checks = [
        ("ollama reachable", check_ollama_reachable),
        ("ollama models present", check_models_present),
        ("hf models cached (offline)", check_hf_models_cached),
    ]
    if not args.no_tests:
        checks.append(("pytest", check_pytest))
    checks.append(("end-to-end query", check_end_to_end))

    all_ok = True
    for name, fn in checks:
        try:
            ok, msg = fn()
        except Exception as exc:  # noqa: BLE001
            ok, msg = False, f"crashed: {exc}"
        all_ok = all_ok and ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}")
    print("\n" + ("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
