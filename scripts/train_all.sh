#!/usr/bin/env bash
# Train the four neural models used in the paper-style write-up.
# GPT and LSTM see exactly the same number of tokens per task.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/training
python3 -m nexttok.train --kind protein --model gpt  --steps 1200 --eval_every 100 --batch 32 | tee results/training/protein_gpt.log
python3 -m nexttok.train --kind protein --model lstm --steps 1200 --eval_every 100 --batch 32 | tee results/training/protein_lstm.log
python3 -m nexttok.train --kind dna     --model gpt  --steps 800 --eval_every 100 --batch 32 | tee results/training/dna_gpt.log
python3 -m nexttok.train --kind dna     --model lstm --steps 800 --eval_every 100 --batch 32 | tee results/training/dna_lstm.log
