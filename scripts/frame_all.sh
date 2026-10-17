#!/usr/bin/env bash
# Reading-frame analysis: GeneMark-style baselines, linear probes, and the codon
# tables for every genome. Needs scripts/train_genomes.sh (or its checkpoints in
# runs/) for the second and third genomes. One process at a time keeps memory low.
set -euo pipefail
cd "$(dirname "$0")/.."
for g in ecoli bsubtilis scoelicolor; do
  python3 -m nexttok.frame markov  --genome $g   # CPU: n-gram ladder, frame Markov chains
  python3 -m nexttok.frame neural  --genome $g   # GPU: LSTM and GPT on the same targets
  python3 -m nexttok.frame summary --genome $g
  python3 -m nexttok.probe --genome $g
done
python3 -m nexttok.frame_report
