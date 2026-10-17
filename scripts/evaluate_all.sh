#!/usr/bin/env bash
# Run every evaluation part, then build figures and results/summary.md.
set -euo pipefail
cd "$(dirname "$0")/.."
for kind in protein dna; do
  python3 -m nexttok.evaluate ngram   --kind $kind
  python3 -m nexttok.evaluate neural  --kind $kind
  python3 -m nexttok.evaluate context --kind $kind
  python3 -m nexttok.evaluate cache   --kind $kind
done
python3 -m nexttok.evaluate audit    --kind protein
python3 -m nexttok.evaluate generate --kind protein
python3 -m nexttok.report
