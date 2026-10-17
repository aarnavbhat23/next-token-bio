#!/usr/bin/env bash
# Second and third genomes for the reading-frame analysis: download, label, and
# train the DNA LSTM and GPT with the E. coli budget (800 x 32 x 512 tokens).
# Checkpoints and logs go to runs/<genome>/; the E. coli files are not touched.
set -euo pipefail
cd "$(dirname "$0")/.."
for g in bsubtilis scoelicolor; do
  [ -f data/raw/$(python3 -c "from nexttok.genome import GENOMES; print(GENOMES['$g'].stem)").fasta ] \
    || python3 -m nexttok.genome download --genome $g
  python3 -m nexttok.genome prepare --genome $g
  mkdir -p runs/$g/results/training
  for m in lstm gpt; do
    python3 -m nexttok.genome train --genome $g --model $m | tee runs/$g/results/training/dna_$m.log
  done
done
