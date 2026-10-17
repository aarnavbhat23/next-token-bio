#!/usr/bin/env bash
# Fetch the six ProteinGym v1.3 DMS assays used by nexttok.variants (~43 MB zip, six CSVs kept)
# plus the reference table and the published zero-shot Spearman table (Notin et al., NeurIPS 2023).
set -euo pipefail
cd "$(dirname "$0")/../data" && mkdir -p raw/proteingym && cd raw/proteingym
GH=https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main
curl -sSL -o DMS_substitutions.csv "$GH/reference_files/DMS_substitutions.csv"
curl -sSL -o DMS_substitutions_Spearman_DMS_level.csv \
  "$GH/benchmarks/DMS_zero_shot/substitutions/Spearman/DMS_substitutions_Spearman_DMS_level.csv"
tmp=$(mktemp -d)
curl -sSL --retry 3 -o "$tmp/subs.zip" https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip
for a in BLAT_ECOLX_Jacquier_2013 P53_HUMAN_Kotler_2018 CCDB_ECOLI_Adkar_2012 NUD15_HUMAN_Suiter_2020 \
         RASK_HUMAN_Weng_2022_binding-DARPin_K55 NUSA_ECOLI_Tsuboyama_2023_1WCL; do
  unzip -o -j -q "$tmp/subs.zip" "DMS_ProteinGym_substitutions/$a.csv" -d .
done
rm -rf "$tmp"
