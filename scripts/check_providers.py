"""Audit every configured LLM provider and the evaluation role assignment.

Two things go wrong silently with free-tier providers, and both are expensive to
diagnose later:

  1. A key is present but rejected (a Google Cloud Console key has no free-tier
     quota; only an aistudio.google.com key does). The pipeline then falls back
     to local and the run quietly changes model mid-way.
  2. A pinned model id has been retired. That surfaces as a 404 that looks
     exactly like an outage.

This script asks every provider what it actually accepts, and checks the three
evaluation roles (answerer / judge / question author) are still three different
models — if any two collapse onto one model, faithfulness scores are inflated by
a model grading its own output.

    python scripts/check_providers.py
    python scripts/check_providers.py --list groq     # dump one provider's models

Exits non-zero if a configured provider is broken or the roles are not distinct.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402
import providers  # noqa: E402

OK, BAD, WARN = "✅", "❌", "⚠️ "


def audit_providers() -> tuple[int, int]:
    """Print one line per provider. Returns (n_working, n_broken)."""
    print("Providers")
    print("-" * 72)
    working = broken = 0
    for name, p in providers.REGISTRY.items():
        if not p.available:
            print(f"  {WARN} {name:<11} no key in ${p.key_env} — skipped")
            continue
        try:
            ids = providers.list_models(name)
        except Exception as exc:  # noqa: BLE001
            print(f"  {BAD} {name:<11} {str(exc)[:88]}")
            broken += 1
            continue
        working += 1
        # Compare canonically: Google lists "models/<id>" and OpenRouter appends
        # ":free", so raw string matching reports false failures.
        canon = {providers.canonical_model(i) for i in ids}
        default_ok = (not ids) or (providers.canonical_model(p.default_model) in canon)
        mark = OK if default_ok else WARN
        note = "" if default_ok else f"  (default {p.default_model!r} NOT offered)"
        print(f"  {mark} {name:<11} {len(ids):>3} models{note}")
    return working, broken


def audit_roles() -> bool:
    """Check the three evaluation roles are distinct and their models are real."""
    print("\nEvaluation roles")
    print("-" * 72)
    roles = config.evaluation_roles()
    for role, spec in roles.items():
        print(f"  {role:<13} {spec}")

    distinct, detail = config.roles_are_distinct()
    print(f"\n  {OK if distinct else BAD} self-preference check: {detail}")
    if not distinct:
        print("     Two roles share a model. The judge would be grading its own")
        print("     output or its own phrasing, which inflates faithfulness.")
        print("     Fix: set JUDGE_MODEL / QUESTIONGEN_MODEL to different models.")

    # Validate every entry of every chain — not just the head, and not the
    # legacy single-model settings. A dead entry deep in a chain is invisible
    # until the entries above it are exhausted, which is precisely when a long
    # run is already in progress.
    ok = distinct
    chains = (
        ("answerer", config.GENERATION_CHAIN),
        ("judge", config.JUDGE_CHAIN),
        ("question-gen", config.QUESTIONGEN_CHAIN),
    )
    print()
    for role, chain in chains:
        for i, spec in enumerate(chain):
            provider_name, model = providers.parse_spec(spec)
            if not model:
                continue
            valid, why = providers.validate_model(provider_name, model)
            tier = "primary " if i == 0 else f"fallback{i}"
            print(f"  {OK if valid else BAD} {role:<13} {tier} {spec}: {why}")
            # Only the primary failing is fatal; a dead fallback is a warning,
            # since the chain still has the local model beneath it.
            ok = ok and (valid or i > 0)
    return ok


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Audit LLM providers and roles")
    p.add_argument("--list", default="", metavar="PROVIDER",
                   help="print every model id offered by one provider")
    args = p.parse_args(argv)

    if args.list:
        try:
            for m in providers.list_models(args.list):
                print(m)
        except Exception as exc:  # noqa: BLE001
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0

    print("RepoMind — provider & role audit\n" + "=" * 72)
    working, broken = audit_providers()
    roles_ok = audit_roles()

    print("\n" + "=" * 72)
    print(f"{working} provider(s) working, {broken} broken")
    if broken or not roles_ok:
        print("RESULT: NEEDS ATTENTION")
        return 1
    print("RESULT: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
