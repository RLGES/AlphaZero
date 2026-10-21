#!/usr/bin/env bash
# Launch v2 AlphaZero training on Modal for each suite and return immediately.
# Runs keep going in the cloud; watch them at https://modal.com/apps and fetch
# results with scripts/modal_fetch.sh. Re-running resumes unfinished runs.
#
#   scripts/modal_v2.sh                       # arithmetic, boolean, bitvector
#   scripts/modal_v2.sh boolean               # one suite
#   ROUNDS=30 scripts/modal_v2.sh arithmetic  # cap the rounds (and the cost)
set -euo pipefail
cd "$(dirname "$0")/.."

MODAL="${MODAL:-.venv/bin/modal}"
ROUNDS="${ROUNDS:-60}"
suites=("$@")
[[ ${#suites[@]} -eq 0 ]] && suites=(arithmetic boolean bitvector)

for suite in "${suites[@]}"; do
  echo "== launching ${suite}_v2 (up to $ROUNDS rounds)"
  "$MODAL" run --detach modal_app.py --suite "$suite" --rounds "$ROUNDS" --no-follow
done
echo "Watch: https://modal.com/apps   Fetch: scripts/modal_fetch.sh ${suites[*]}"
