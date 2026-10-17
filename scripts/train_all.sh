#!/usr/bin/env bash
# Train every neural model in the seed sweep: protein GPT and LSTM on 3 family
# splits x 2 seeds, DNA GPT and LSTM on 3 seeds. GPT and LSTM with the same
# split and seed see exactly the same batches. Runs go one at a time: on a
# shared Apple GPU two concurrent runs were no faster in total and needed twice
# the memory. A run with results/training/<run>.json is finished and skipped;
# any other run continues from checkpoints/<run>.last.pt (train --resume), so
# after an interruption just rerun this script. PYTHON picks the interpreter.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/training
PY=${PYTHON:-python3}
PROTEIN_STEPS=${PROTEIN_STEPS:-3600}
DNA_STEPS=${DNA_STEPS:-2400}

run() {  # kind model split_seed seed steps
  local name
  if [ "$1" = protein ]; then name="protein_$2_split$3_seed$4"; else name="dna_$2_seed$4"; fi
  if [ -f "results/training/$name.json" ]; then echo "skip $name (finished)"; return; fi
  if "$PY" -m nexttok.train --kind "$1" --model "$2" --split_seed "$3" --seed "$4" --steps "$5" \
       --eval_every 100 --batch 32 --resume >> "results/training/$name.log" 2>&1; then
    echo "$(date '+%H:%M') done $name"
  else
    echo "$(date '+%H:%M') FAILED $name"
  fi
}

for split in 0 1 2; do for m in gpt lstm; do run protein $m "$split" 0 "$PROTEIN_STEPS"; done; done
for seed in 0 1 2; do for m in gpt lstm; do run dna $m 0 "$seed" "$DNA_STEPS"; done; done
for split in 0 1 2; do for m in gpt lstm; do run protein $m "$split" 1 "$PROTEIN_STEPS"; done; done
