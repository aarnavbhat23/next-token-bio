"""Build the protein and DNA datasets with leakage-aware splits.

Protein (UniProtKB/Swiss-Prot, reviewed, length 50-512):
  * sequences with non-standard residues are dropped, exact duplicates removed
  * sequences without a family annotation are dropped entirely, so no
    unlabelled homolog of a held-out family can sneak into training
  * family key = first clause of the "Protein families" field (so
    "Protein kinase superfamily, Ser/Thr protein kinase family" groups at
    the superfamily level)
  * whole families are hashed into train (80%) / val (10%) / test_novel (10%);
    --split_seed salts the hash, so each seed holds out a different set of
    families (seed 0 is the unsalted hash of the first release)
  * test_seen = 3% of sequences drawn at random from *training* families

  test_seen vs test_novel is the central contrast: same distribution of
  sequence lengths and organisms, but only one of them has close homologs
  in the training data.

DNA (E. coli K-12 MG1655, NC_000913.3):
  * genome cut into 50 kb blocks; block % 10 == 9 -> test, == 4 -> val
  * every position gets a label: non-coding, + strand CDS codon position 1-3,
    - strand CDS codon position 1-3, or overlapping genes
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from .data import PROC
from .vocab import BOS, EOS, encode_table

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

DNA_BLOCK = 50_000
DNA_LABELS = ["noncoding", "+cds_pos1", "+cds_pos2", "+cds_pos3",
              "-cds_pos1", "-cds_pos2", "-cds_pos3", "overlap"]


def family_key(field: str) -> str:
    return re.split(r"[,;]", field)[0].strip().lower()


def family_bucket(key: str, split_seed: int = 0) -> str:
    salt = f"{split_seed}:" if split_seed else ""  # seed 0 keeps the first release's split
    h = int(hashlib.sha1((salt + key).encode()).hexdigest(), 16) % 100
    return "train" if h < 80 else ("val" if h < 90 else "test_novel")


def pack(seqs: list[np.ndarray]) -> dict[str, np.ndarray]:
    lengths = np.array([len(s) for s in seqs], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths + 2)[:-1]]).astype(np.int64)
    tokens = np.empty(int((lengths + 2).sum()), dtype=np.uint8)
    for off, s in zip(offsets, seqs):
        tokens[off] = BOS
        tokens[off + 1: off + 1 + len(s)] = s
        tokens[off + 1 + len(s)] = EOS
    return {"tokens": tokens, "offsets": offsets, "lengths": lengths}


def prepare_protein(seed: int = 0, split_seed: int = 0) -> dict:
    table = encode_table("protein")
    seen: set[str] = set()
    rows = {"train": [], "val": [], "test_novel": []}
    stats = {"rows": 0, "nonstandard": 0, "duplicate": 0, "no_family": 0}
    with gzip.open(RAW / "swissprot_50_512.tsv.gz", "rt") as fh:
        next(fh)
        for line in fh:
            stats["rows"] += 1
            acc, _org, fam, _len, seq = line.rstrip("\n").split("\t")
            enc = table[np.frombuffer(seq.encode(), dtype=np.uint8)]
            if (enc == 255).any():
                stats["nonstandard"] += 1
                continue
            if seq in seen:
                stats["duplicate"] += 1
                continue
            seen.add(seq)
            if not fam:
                stats["no_family"] += 1
                continue
            key = family_key(fam)
            rows[family_bucket(key, split_seed)].append((acc, key, enc))

    rng = np.random.default_rng(seed)
    train = rows.pop("train")
    is_seen_test = rng.random(len(train)) < 0.03
    rows["test_seen"] = [r for r, t in zip(train, is_seen_test) if t]
    rows["train"] = [r for r, t in zip(train, is_seen_test) if not t]

    fam_index = {k: i for i, k in enumerate(sorted({r[1] for v in rows.values() for r in v}))}
    summary = {"split_seed": split_seed, "filter": stats, "splits": {}}
    out = PROC / f"split{split_seed}"
    out.mkdir(parents=True, exist_ok=True)
    for split, rs in rows.items():
        order = rng.permutation(len(rs))
        rs = [rs[i] for i in order]
        packed = pack([r[2] for r in rs])
        packed["family"] = np.array([fam_index[r[1]] for r in rs], dtype=np.int32)
        np.savez(out / f"protein_{split}.npz", **packed)
        (out / f"protein_{split}_accessions.txt").write_text("\n".join(r[0] for r in rs) + "\n")
        summary["splits"][split] = {
            "sequences": len(rs),
            "residues": int(packed["lengths"].sum()),
            "families": int(len(set(packed["family"].tolist()))),
        }
    train_f = {r[1] for r in rows["train"]}
    for split in ("val", "test_novel"):
        assert not train_f & {r[1] for r in rows[split]}, f"family leak into {split}"
    (out / "protein_families.json").write_text(json.dumps(sorted(fam_index, key=fam_index.get)))
    return summary


def prepare_dna() -> dict:
    fasta = (RAW / "ecoli_k12.fasta").read_text().splitlines()
    seq = "".join(l.strip() for l in fasta if not l.startswith(">"))
    enc = encode_table("dna")[np.frombuffer(seq.encode(), dtype=np.uint8)]
    assert not (enc == 255).any()
    n = len(enc)

    plus = np.zeros(n, dtype=np.int8)   # codon position 1..3 on + strand, 0 = none
    minus = np.zeros(n, dtype=np.int8)
    cover = np.zeros(n, dtype=np.int8)
    used = skipped = 0
    for line in (RAW / "ecoli_k12.gff3").read_text().splitlines():
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

    block = np.arange(n) // DNA_BLOCK
    split = np.where(block % 10 == 9, 2, np.where(block % 10 == 4, 1, 0)).astype(np.int8)
    np.savez(PROC / "dna_ecoli.npz", tokens=enc, labels=labels, split=split, block=block.astype(np.int32))
    names = ["train", "val", "test"]
    return {
        "genome_length": n,
        "cds_used": used,
        "cds_skipped": skipped,
        "label_fraction": {DNA_LABELS[i]: float((labels == i).mean()) for i in range(len(DNA_LABELS))},
        "split_bases": {names[i]: int((split == i).sum()) for i in range(3)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, help="draw of the seen-family test set")
    ap.add_argument("--split_seed", type=int, nargs="+", default=[0],
                    help="family split(s); each is written to data/processed/split<N>/")
    args = ap.parse_args()
    PROC.mkdir(parents=True, exist_ok=True)
    summary = {"dna": prepare_dna(), "protein": {}}
    for s in args.split_seed:
        summary["protein"][f"split{s}"] = prepare_protein(args.seed, s)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "data_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
