#!/usr/bin/env bash
# Download v2 results from the Modal volume: best network, latest network and
# per-round metrics, into checkpoints/<suite>_v2/.
#
#   scripts/modal_fetch.sh                  # all three suites
#   scripts/modal_fetch.sh boolean
set -euo pipefail
cd "$(dirname "$0")/.."

MODAL="${MODAL:-.venv/bin/modal}"
suites=("$@")
[[ ${#suites[@]} -eq 0 ]] && suites=(arithmetic boolean bitvector)

for suite in "${suites[@]}"; do
  run="${suite}_v2"
  mkdir -p "checkpoints/$run"
  for file in best.pt latest.pt metrics.jsonl; do
    "$MODAL" volume get --force rewritezero-checkpoints "$run/$file" "checkpoints/$run/$file" \
      || echo "  ($run/$file not there yet)"
  done
  if [[ -f "checkpoints/$run/metrics.jsonl" ]]; then
    echo "== $run: $(wc -l < "checkpoints/$run/metrics.jsonl" | tr -d ' ') rounds; last:"
    tail -1 "checkpoints/$run/metrics.jsonl"
  fi
done
