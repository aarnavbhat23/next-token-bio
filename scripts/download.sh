#!/usr/bin/env bash
# Fetch the two public datasets (~80 MB total).
set -euo pipefail
cd "$(dirname "$0")/../data" && mkdir -p raw && cd raw
# UniProtKB/Swiss-Prot reviewed entries, length 50-512, with family annotation
curl -sSL -o swissprot_50_512.tsv.gz \
  "https://rest.uniprot.org/uniprotkb/stream?compressed=true&format=tsv&fields=accession,organism_id,protein_families,length,sequence&query=%28reviewed%3Atrue%29%20AND%20%28length%3A%5B50%20TO%20512%5D%29"
# E. coli K-12 MG1655 genome and RefSeq annotation
curl -sSL -o ecoli_k12.fasta "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_000913.3&rettype=fasta&retmode=text"
curl -sSL -o ecoli_k12.gff3 "https://www.ncbi.nlm.nih.gov/sviewer/viewer.cgi?db=nuccore&report=gff3&id=NC_000913.3"
