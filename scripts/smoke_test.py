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


# --------------------------------------------------------------------------- #
# Durability checks — the things that rot between now and the demo
# --------------------------------------------------------------------------- #
def check_python_version() -> tuple[bool, str]:
    """A macOS or Homebrew upgrade can move python3 out from under the venv."""
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 11):
        return False, f"Python {major}.{minor} — the pinned wheels target 3.11+"
    return True, f"Python {major}.{minor}.{sys.version_info[2]}"


def check_github_token() -> tuple[bool, str]:
    """Validity, scope and — the one with a real deadline — expiry.

    A GitHub token that expires mid-gap is the single most likely scheduled
    failure: ingestion silently loses pull requests (GraphQL needs auth) long
    before anyone notices.
    """
    import requests
    if not config.GITHUB_TOKEN:
        return False, "GITHUB_TOKEN not set — PRs cannot be ingested at all"
    try:
        r = requests.get("https://api.github.com/user",
                         headers={"Authorization": f"Bearer {config.GITHUB_TOKEN}"},
                         timeout=15)
    except Exception as exc:  # noqa: BLE001
        return False, f"could not reach GitHub: {exc}"
    if r.status_code == 401:
        return False, "GITHUB_TOKEN rejected (401) — expired or revoked"
    if r.status_code >= 400:
        return False, f"GitHub returned {r.status_code}"

    expiry = r.headers.get("github-authentication-token-expiration")
    limit = r.headers.get("x-ratelimit-limit", "?")
    if not expiry:
        return True, f"valid, no expiry set, {limit} req/hr"
    # Warn well before it lapses — renewing after the fact means re-indexing.
    from datetime import datetime, timezone
    try:
        when = datetime.fromisoformat(expiry.replace(" UTC", "+00:00").strip())
        days = (when - datetime.now(timezone.utc)).days
        if days < 0:
            return False, f"token EXPIRED on {expiry}"
        if days < 60:
            return False, (f"token expires in {days} days ({expiry}) — renew now; "
                           f"a lapse silently drops PRs from ingestion")
        return True, f"valid until {expiry} ({days} days), {limit} req/hr"
    except ValueError:
        return True, f"valid, expires {expiry}, {limit} req/hr"


def check_pinned_dependencies() -> tuple[bool, str]:
    """Installed versions must still match requirements.txt exactly.

    The whole "runs unattended for a year" claim rests on these pins. A stray
    `pip install` in the venv breaks reproducibility without any visible symptom.
    """
    import re
    from importlib.metadata import PackageNotFoundError, version

    req = _ROOT / "requirements.txt"
    if not req.exists():
        return False, "requirements.txt missing"
    drift, missing = [], []
    for line in req.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;]+)$", line)
        if not m:
            continue
        name, want = m.group(1), m.group(2)
        try:
            have = version(name)
        except PackageNotFoundError:
            missing.append(name)
            continue
        if have != want:
            drift.append(f"{name} {have}!={want}")
    if missing or drift:
        parts = []
        if missing:
            parts.append(f"not installed: {', '.join(missing[:5])}")
        if drift:
            parts.append(f"version drift: {', '.join(drift[:5])}")
        return False, "; ".join(parts)
    return True, "all pinned dependencies match requirements.txt"


def check_indexed_repositories() -> tuple[bool, str]:
    """Every index must still load — Chroma's on-disk format can move."""
    from core import registry
    from core.context import RepositoryContext
    from core.repo_url import parse_repo_url
    from index.vector_store import load_bm25_payload, load_collection

    entries = registry.list_repositories()
    if not entries:
        return False, "no indexed repositories"
    broken = []
    for e in entries:
        if not e.reusable:
            broken.append(f"{e.full_name} (stale: {e.reuse_reason})")
            continue
        try:
            ctx = RepositoryContext.for_ref(parse_repo_url(e.full_name))
            load_collection(ctx).count()
            load_bm25_payload(ctx)
        except Exception as exc:  # noqa: BLE001
            broken.append(f"{e.full_name} ({str(exc)[:50]})")
    if broken:
        return False, f"{len(broken)} unusable: {'; '.join(broken)}"
    return True, f"{len(entries)} repositories load cleanly"


def check_offline_restore() -> tuple[bool, str]:
    """The wheelhouse is the answer to 'PyPI removed a pinned version'."""
    wh = _ROOT / "wheelhouse"
    if not wh.exists():
        return True, "no wheelhouse (fine, but a clean rebuild then needs PyPI)"
    wheels = list(wh.glob("*.whl"))
    hf = wh / "hf_cache"
    bits = [f"{len(wheels)} wheels"]
    bits.append("HF model cache present" if hf.exists() else "NO HF cache")
    return (len(wheels) > 0), "offline restore: " + ", ".join(bits)


def check_disk_space() -> tuple[bool, str]:
    """Indexes plus model caches are tens of GB; a full disk fails obscurely."""
    import shutil
    free_gb = shutil.disk_usage(_ROOT).free / 1e9
    if free_gb < 5:
        return False, f"only {free_gb:.1f} GB free — indexing will fail"
    return True, f"{free_gb:.1f} GB free"


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
        # Environment — breaks from the outside (OS upgrades, disk, pip drift)
        ("python version", check_python_version),
        ("disk space", check_disk_space),
        ("pinned dependencies", check_pinned_dependencies),
        # Models — must resolve locally, with no network
        ("ollama reachable", check_ollama_reachable),
        ("ollama models present", check_models_present),
        ("hf models cached (offline)", check_hf_models_cached),
        ("offline restore", check_offline_restore),
        # Credentials — the one with a calendar deadline
        ("github token", check_github_token),
        # Data — indexes must still load
        ("indexed repositories", check_indexed_repositories),
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
