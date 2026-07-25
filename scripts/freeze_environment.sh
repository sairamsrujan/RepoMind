#!/usr/bin/env bash
# Phase G3: build an offline wheelhouse + model-cache archive, and record an
# exact environment snapshot into ENVIRONMENT.md. See ENVIRONMENT.md for the
# matching restore procedure. Safe to re-run.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-.venv/bin/python}"

echo "==> [1/3] Building offline wheelhouse (pip download)..."
mkdir -p wheelhouse
"$PY" -m pip download -r requirements.txt -d wheelhouse/

echo "==> [2/3] Archiving HuggingFace cache for reranker + NLI..."
HF_HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"
mkdir -p wheelhouse/hf_cache
for pat in \
  "models--BAAI--bge-reranker-v2-m3" \
  "models--cross-encoder--ms-marco-MiniLM-L-6-v2" \
  "models--cross-encoder--nli-deberta-v3-base"; do
  if [ -d "$HF_HUB/$pat" ]; then
    cp -R "$HF_HUB/$pat" "wheelhouse/hf_cache/" && echo "    archived $pat"
  else
    echo "    (skip, not cached) $pat"
  fi
done

echo "==> [3/3] Recording environment snapshot into ENVIRONMENT.md..."
SNAP="$(mktemp)"
{
  echo "## Snapshot"
  echo
  echo "_Generated $(date -u +%Y-%m-%dT%H:%M:%SZ)_"
  echo
  echo '```'
  echo "Python: $($PY --version 2>&1)"
  echo "macOS:  $(sw_vers 2>/dev/null | tr '\n' ' ' || echo 'n/a')"
  echo
  echo "ollama list:"
  ollama list 2>/dev/null || echo "  (ollama not running)"
  echo
  echo "pip freeze:"
  "$PY" -m pip freeze | sort
  echo '```'
} > "$SNAP"

# Rebuild ENVIRONMENT.md: keep everything outside the SNAPSHOT markers, replace
# the content between them with the fresh snapshot.
awk -v snapfile="$SNAP" '
  /<!-- SNAPSHOT:BEGIN -->/ { print; while ((getline l < snapfile) > 0) print l; skip=1; next }
  /<!-- SNAPSHOT:END -->/   { skip=0 }
  skip != 1 { print }
' ENVIRONMENT.md > ENVIRONMENT.md.tmp && mv ENVIRONMENT.md.tmp ENVIRONMENT.md
rm -f "$SNAP"

echo "==> Done. wheelhouse/ built and ENVIRONMENT.md snapshot updated."
echo "    Keep wheelhouse/ on external storage (it is gitignored)."
