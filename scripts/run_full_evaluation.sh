#!/usr/bin/env bash
#
# Run the complete evaluation suite, in dependency order, resumably.
#
# Run this when the provider quotas have reset (they are daily). Everything here
# is checkpointed: eval/run.py writes rows_partial.jsonl after every question, so
# re-running this exact script after any interruption resumes rather than
# restarting. Nothing is recomputed that already succeeded.
#
#   bash scripts/run_full_evaluation.sh              # everything
#   bash scripts/run_full_evaluation.sh abstention   # just the abstention sets
#   bash scripts/run_full_evaluation.sh mixed        # just the mixed-category sets
#   bash scripts/run_full_evaluation.sh ablation     # just the ablation table
#
# IMPORTANT (HANDOFF.md 3.5): run ONE heavy job at a time on a 16 GB machine.
# Do not run the Streamlit app or pytest while this is going.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
STAGE="${1:-all}"

REPOS=(
  "pallets/click:pallets_click"
  "psf/requests:psf_requests"
  "psf/black:psf_black"
  "pydantic/pydantic:pydantic_pydantic"
  "fastapi/fastapi:fastapi_fastapi"
  "acme/widgets:acme_widgets"
)

banner () { echo; echo "=============== $* ==============="; }

# --------------------------------------------------------------------------- #
# 0. Refuse to start against broken providers — otherwise the whole run
#    silently degrades to the local model and the numbers mean something else.
# --------------------------------------------------------------------------- #
banner "provider + role audit"
if ! $PY scripts/check_providers.py; then
  echo
  echo "Providers or roles are not healthy. Fix them before running a long"
  echo "evaluation, or the results will be a mix of models. Continuing anyway"
  echo "is valid only if you accept local-model provenance in the report."
  read -r -p "Continue regardless? [y/N] " reply
  [[ "$reply" == "y" || "$reply" == "Y" ]] || exit 1
fi

# --------------------------------------------------------------------------- #
# 1. Abstention sets — the headline metric. Unanswerable questions skip the
#    judge entirely, so this stage costs NO judge quota.
# --------------------------------------------------------------------------- #
if [[ "$STAGE" == "all" || "$STAGE" == "abstention" ]]; then
  for entry in "${REPOS[@]}"; do
    repo="${entry%%:*}"; slug="${entry##*:}"
    banner "abstention — $repo"
    $PY -m eval.run --repo "$repo" \
      --dataset "eval/datasets/${slug}_abstention.jsonl" \
      --out "results/abstention-${slug}"
  done
fi

# --------------------------------------------------------------------------- #
# 2. Mixed-category golden sets. The two newest repositories have no mixed set
#    yet; generate it first (question generation, not judging, so it uses the
#    QUESTIONGEN chain).
# --------------------------------------------------------------------------- #
if [[ "$STAGE" == "all" || "$STAGE" == "mixed" ]]; then
  for entry in "psf/black:psf_black" "pydantic/pydantic:pydantic_pydantic"; do
    repo="${entry%%:*}"; slug="${entry##*:}"
    if [[ ! -f "eval/datasets/${slug}.jsonl" ]]; then
      banner "generating golden set — $repo"
      $PY -m eval.generate_golden_set --repo "$repo" --n 50
    fi
  done

  for entry in "${REPOS[@]}"; do
    repo="${entry%%:*}"; slug="${entry##*:}"
    [[ -f "eval/datasets/${slug}.jsonl" ]] || continue
    banner "evaluation — $repo"
    $PY -m eval.run --repo "$repo" \
      --dataset "eval/datasets/${slug}.jsonl" \
      --out "results/eval-${slug}"
  done
fi

# --------------------------------------------------------------------------- #
# 3. Ablation. Cost is (configs x questions), so it is scoped with --limit.
#    Raise the limit only if the quota clearly allows it.
# --------------------------------------------------------------------------- #
if [[ "$STAGE" == "all" || "$STAGE" == "ablation" ]]; then
  banner "ablation (8 configs)"
  $PY -m eval.ablation \
    --repos pallets/click,psf/requests \
    --limit 20 \
    --out results/ablation-multi
fi

banner "done"
echo "Reports:      results/*/report.txt"
echo "Ablation:     results/ablation-multi/ablation.csv"
echo "Summarise:    $PY scripts/summarise_metrics.py"
