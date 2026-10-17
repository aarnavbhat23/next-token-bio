#!/usr/bin/env bash
# Run every evaluation part for every family split and training seed, then
# build the figures, results/intervals.json and results/summary.md.
# Needs the full seed sweep (scripts/train_all.sh). PYTHON picks the interpreter.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
for split in 0 1 2; do
  for part in ngram cache audit; do
    "$PY" -m nexttok.evaluate $part --kind protein --split_seed $split
  done
  for part in neural context; do
    "$PY" -m nexttok.evaluate $part --kind protein --split_seed $split --seeds 0 1
  done
done
"$PY" -m nexttok.evaluate ngram   --kind dna
"$PY" -m nexttok.evaluate cache   --kind dna
"$PY" -m nexttok.evaluate neural  --kind dna --seeds 0 1 2
"$PY" -m nexttok.evaluate context --kind dna --seeds 0 1 2
"$PY" -m nexttok.evaluate generate --kind protein --split_seed 0 --seeds 0
"$PY" -m nexttok.report
