"""Run the DNA half of the pipeline on other bacterial genomes.

The first release uses one genome, E. coli K-12. This module adds others
without touching the E. coli code path:

  python -m nexttok.genome download --genome bsubtilis
  python -m nexttok.genome prepare  --genome bsubtilis
  python -m nexttok.genome train    --genome bsubtilis --model lstm

`prepare` applies the labelling and 50 kb block split of
nexttok.prepare.prepare_dna; on E. coli it reproduces data/processed/dna_ecoli.npz
array for array. Other genomes go to data/processed/dna_<genome>.npz.

`train` runs the unchanged nexttok.train.main with data.load_dna pointed at that
file and the output root moved to runs/<genome>/, so the new checkpoints and
training logs never overwrite the E. coli ones. Evaluation is in nexttok.frame.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import data, evaluate, train
from .prepare import DNA_BLOCK, DNA_LABELS
from .vocab import encode_table

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RUNS = ROOT / "runs"


@dataclass(frozen=True)
class Genome:
    accession: str
    stem: str      # raw files are data/raw/<stem>.fasta and .gff3
    species: str


GENOMES = {
    "ecoli": Genome("NC_000913.3", "ecoli_k12", "Escherichia coli K-12 MG1655"),
    "bsubtilis": Genome("NC_000964.3", "bsubtilis_168", "Bacillus subtilis 168"),
    "scoelicolor": Genome("NC_003888.3", "scoelicolor_a3_2", "Streptomyces coelicolor A3(2)"),
}


def processed_path(name: str) -> Path:
    return data.PROC / f"dna_{name}.npz"


def run_root(name: str) -> Path:
    """Where a genome's checkpoints and training logs live (the repo root for E. coli)."""
    return ROOT if name == "ecoli" else RUNS / name


def checkpoint(name: str, model: str) -> Path:
    """The 800-step checkpoint for this genome (first-release naming, else seed-0 naming)."""
    ck = run_root(name) / "checkpoints"
    for p in (ck / f"dna_{model}.pt", ck / f"dna_{model}_seed0.pt"):
        if p.exists():
            return p
    raise FileNotFoundError(f"no {model} checkpoint for {name} in {ck}")


def load(name: str) -> dict[str, np.ndarray]:
    return dict(np.load(processed_path(name)))


# ----------------------------------------------------------------- download
def download(name: str) -> None:
    g = GENOMES[name]
    RAW.mkdir(parents=True, exist_ok=True)
    urls = {
        "fasta": f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id={g.accession}&rettype=fasta&retmode=text",
        "gff3": f"https://www.ncbi.nlm.nih.gov/sviewer/viewer.cgi?db=nuccore&report=gff3&id={g.accession}",
    }
    for ext, url in urls.items():
        with urllib.request.urlopen(url, timeout=300) as r:
            (RAW / f"{g.stem}.{ext}").write_bytes(r.read())
        time.sleep(1)  # NCBI asks for at most 3 requests a second


# ----------------------------------------------------------------- labelling
def label_genome(seq: str, gff: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Token ids and per-base labels (index into DNA_LABELS), as in prepare_dna.

    + strand codon position counts from the CDS start; - strand codon position
    counts from the CDS end, so reading the + strand 5'->3' through a - strand
    gene visits its codon positions 3, 2, 1, 3, 2, 1, ... Bases in two CDS are
    labelled "overlap". Pseudogenes and CDS whose length is not a multiple of
    three are skipped.
    """
    enc = encode_table("dna")[np.frombuffer(seq.encode(), dtype=np.uint8)]
    assert not (enc == 255).any(), "non-ACGT symbol in genome"
    n = len(enc)
    plus = np.zeros(n, dtype=np.int8)
    minus = np.zeros(n, dtype=np.int8)
    cover = np.zeros(n, dtype=np.int8)
    used = skipped = 0
    for line in gff.splitlines():
        if line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) < 9 or f[2] != "CDS":
            continue
        start, end, strand = int(f[3]) - 1, int(f[4]), f[6]  # 0-based half open
        if "pseudo=true" in f[8] or (end - start) % 3 or "Parent=gene-" not in f[8]:
            skipped += 1
            continue
        idx = np.arange(start, end)
        if strand == "+":
            plus[idx] = (idx - start) % 3 + 1
        else:
            minus[idx] = (end - 1 - idx) % 3 + 1
        cover[idx] += 1
        used += 1
    labels = np.zeros(n, dtype=np.int8)
    labels[plus > 0] = plus[plus > 0]
    labels[minus > 0] = minus[minus > 0] + 3
    labels[cover > 1] = 7
    return enc, labels, {"cds_used": used, "cds_skipped": skipped}


def block_split(n: int) -> tuple[np.ndarray, np.ndarray]:
    """50 kb blocks; block % 10 == 9 is test (2), == 4 is validation (1)."""
    block = np.arange(n) // DNA_BLOCK
    split = np.where(block % 10 == 9, 2, np.where(block % 10 == 4, 1, 0)).astype(np.int8)
    return split, block.astype(np.int32)


def prepare(name: str) -> dict:
    g = GENOMES[name]
    fasta = (RAW / f"{g.stem}.fasta").read_text().splitlines()
    seq = "".join(l.strip() for l in fasta if not l.startswith(">"))
    enc, labels, stats = label_genome(seq, (RAW / f"{g.stem}.gff3").read_text())
    split, block = block_split(len(enc))
    data.PROC.mkdir(parents=True, exist_ok=True)
    np.savez(processed_path(name), tokens=enc, labels=labels, split=split, block=block)
    sym = enc.astype(np.int64)
    names = ["train", "val", "test"]
    return {
        "genome": name, "accession": g.accession, "species": g.species,
        "genome_length": len(enc), **stats,
        "gc": float(np.isin(sym, encode_table("dna")[np.frombuffer(b"CG", np.uint8)]).mean()),
        "label_fraction": {DNA_LABELS[i]: float((labels == i).mean()) for i in range(len(DNA_LABELS))},
        "split_bases": {names[i]: int((split == i).sum()) for i in range(3)},
    }


# ----------------------------------------------------------------- training
@contextmanager
def use_genome(name: str):
    """Point the unchanged DNA pipeline at one genome.

    data.load_dna (and the reference evaluate imported) read that genome's
    processed file, and train.main writes under runs/<genome>/.
    """
    if name == "ecoli":
        yield
        return
    path = processed_path(name)
    saved = data.load_dna, evaluate.load_dna, train.ROOT
    data.load_dna = evaluate.load_dna = lambda: dict(np.load(path))
    train.ROOT = run_root(name)
    try:
        yield
    finally:
        data.load_dna, evaluate.load_dna, train.ROOT = saved


def train_model(name: str, model: str, steps: int, batch: int, eval_every: int) -> None:
    """Same arguments as scripts/train_all.sh used for E. coli DNA."""
    argv = sys.argv
    sys.argv = ["nexttok.train", "--kind", "dna", "--model", model, "--steps", str(steps),
                "--eval_every", str(eval_every), "--batch", str(batch)]
    try:
        with use_genome(name):
            train.main()
    finally:
        sys.argv = argv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["download", "prepare", "train"])
    ap.add_argument("--genome", choices=sorted(GENOMES), required=True)
    ap.add_argument("--model", choices=["gpt", "lstm"])
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval_every", type=int, default=100)
    args = ap.parse_args()
    if args.part == "download":
        download(args.genome)
    elif args.part == "prepare":
        summary = prepare(args.genome)
        out = run_root(args.genome) / "results" / f"data_summary_{args.genome}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
    else:
        if args.genome == "ecoli":
            raise SystemExit("E. coli models are trained by scripts/train_all.sh")
        train_model(args.genome, args.model, args.steps, args.batch, args.eval_every)


if __name__ == "__main__":
    main()
