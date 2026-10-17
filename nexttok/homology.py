"""A stricter held-out split: drop held-out proteins with a 30%-identity relative in training.

UniProt family labels are not a homology guarantee. Two MMseqs2 filters
(Steinegger & Soding, 2017) are applied to the held-out families:

  cluster  train + test_novel clustered together at 30% identity and 80%
           coverage of both sequences (easy-cluster); a held-out protein is
           dropped if its cluster also contains a training protein.
  search   the evaluation subset searched against training at the highest
           sensitivity (-s 7.5); dropped if any hit has >= 30% identity over
           >= 50% of the held-out protein (catches shared domains).

The existing checkpoints are then re-scored on the evaluation subset (the same
3,000 held-out proteins as evaluate.py) with and without the dropped proteins.

  python -m nexttok.homology cluster   # ~5 min on 6 threads
  python -m nexttok.homology search    # ~15 min on 6 threads
  python -m nexttok.homology score     # writes results/homology_protein.json

Needs the `mmseqs` binary (brew install mmseqs2 / conda install -c bioconda mmseqs2).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import ngram
from .data import PROC, load_protein
from .seqscore import ROOT, load_checkpoint, neural_logp, ngram_logp, to_str

MM = PROC / "mmseqs"
RES = ROOT / "results"
MIN_ID, COV = 0.3, 0.8           # cluster filter
SEARCH_ID, SEARCH_QCOV = 0.3, 0.5  # search filter
N_EVAL = 3000                     # same evaluation subset as evaluate.py
ORDERS = (1, 3, 5)  # 5-gram: the strongest memoriser that fits in ~30 MB


def write_fasta(path: Path, parts: dict[str, np.ndarray | None]) -> int:
    """FASTA with ids "<split>|<row>"; `parts` maps split -> rows to write (None = all)."""
    n = 0
    with open(path, "w") as fh:
        for split, rows in parts.items():
            d = load_protein(split)
            rows = range(len(d["offsets"])) if rows is None else rows
            for i in rows:
                o, L = d["offsets"][i], d["lengths"][i]
                fh.write(f">{split}|{i}\n{to_str(d['tokens'][o + 1: o + 1 + L])}\n")
                n += 1
    return n


def mmseqs(args: list[str], cwd: str) -> str:
    exe = shutil.which("mmseqs")
    if exe is None:
        raise SystemExit("mmseqs not found: brew install mmseqs2 (or conda install -c bioconda mmseqs2)")
    subprocess.run([exe] + args, check=True, cwd=cwd)
    return subprocess.run([exe, "version"], capture_output=True, text=True).stdout.strip()


def run_cluster(threads: int) -> dict:
    MM.mkdir(parents=True, exist_ok=True)
    # mmseqs' workflow scripts break on paths with spaces, so it works in a plain temp dir
    with tempfile.TemporaryDirectory(prefix="mmseqs_") as tmp:
        n = write_fasta(Path(tmp) / "all.fasta", {"train": None, "test_novel": None})
        args = ["easy-cluster", "all.fasta", "c30", "tmp", "--min-seq-id", str(MIN_ID),
                "-c", str(COV), "--cov-mode", "0", "--threads", str(threads)]
        version = mmseqs(args, tmp)
        shutil.copy(Path(tmp) / "c30_cluster.tsv", MM / "c30_cluster.tsv")
    meta = {"command": " ".join(["mmseqs"] + args), "mmseqs_version": version, "sequences": n}
    (MM / "c30_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def run_search(threads: int) -> dict:
    MM.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mmseqs_") as tmp:
        write_fasta(Path(tmp) / "q.fasta", {"test_novel": eval_indices("test_novel")})
        write_fasta(Path(tmp) / "t.fasta", {"train": None})
        args = ["easy-search", "q.fasta", "t.fasta", "hits.m8", "tmp", "-s", "7.5", "-e", "1e-3",
                "--min-seq-id", str(SEARCH_ID), "-c", str(SEARCH_QCOV), "--cov-mode", "2",
                "--format-output", "query,target,fident,qcov,tcov,evalue,bits", "--threads", str(threads)]
        version = mmseqs(args, tmp)
        shutil.copy(Path(tmp) / "hits.m8", MM / "s30_hits.m8")
    meta = {"command": " ".join(["mmseqs"] + args), "mmseqs_version": version}
    (MM / "s30_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def novel_sharing_train(rows) -> set[int]:
    """Indices of test_novel proteins whose cluster contains at least one training protein.

    `rows` are (representative, member) pairs as in mmseqs `*_cluster.tsv`,
    with ids of the form "<split>|<index>".
    """
    members = defaultdict(list)
    for rep, mem in rows:
        members[rep].append(mem)
    out = set()
    for mem in members.values():
        if any(m.startswith("train|") for m in mem):
            out.update(int(m.split("|")[1]) for m in mem if m.startswith("test_novel|"))
    return out


def novel_with_hits(rows, min_id: float = SEARCH_ID, min_qcov: float = SEARCH_QCOV) -> set[int]:
    """Indices of test_novel queries with a training hit passing both thresholds."""
    out = set()
    for q, t, fident, qcov, *_ in rows:
        if t.startswith("train|") and float(fident) >= min_id and float(qcov) >= min_qcov:
            out.add(int(q.split("|")[1]))
    return out


def read_tsv(path: Path):
    with open(path) as fh:
        for line in fh:
            yield line.rstrip("\n").split("\t")


def eval_indices(split: str, n: int = N_EVAL, seed: int = 0) -> np.ndarray:
    """Row indices of `data.protein_subset(split, n, seed)` (same RNG draw, sorted)."""
    m = len(load_protein(split)["offsets"])
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(m, size=min(n, m), replace=False))


def family_ci(per_seq_bits: np.ndarray, per_seq_n: np.ndarray, fam: np.ndarray, B: int = 2000, seed: int = 0):
    """95% interval for residue-weighted bits, resampling whole families."""
    rng = np.random.default_rng(seed)
    u, inv = np.unique(fam, return_inverse=True)
    s = np.bincount(inv, per_seq_bits * per_seq_n, minlength=len(u))
    c = np.bincount(inv, per_seq_n, minlength=len(u))
    draws = rng.integers(0, len(u), (B, len(u)))
    vals = s[draws].sum(1) / c[draws].sum(1)
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def filter_report(drop: np.ndarray, scores: dict, lens: np.ndarray, fam: np.ndarray) -> dict:
    keep = ~drop
    every = np.ones_like(drop)

    def bits(mask, lp):
        if not mask.any():
            return float("nan")
        return float(-np.concatenate([lp[i] for i in np.flatnonzero(mask)]).mean())

    fams = set(fam.tolist())
    out = {"dropped": int(drop.sum()), "kept": int(keep.sum()),
           "residues_dropped": int(lens[drop].sum()), "residues_kept": int(lens[keep].sum()),
           "families_with_a_dropped_protein": len(set(fam[drop].tolist())),
           "families_fully_dropped": len(fams - set(fam[keep].tolist())), "bits": {}}
    for name, lp in scores.items():
        per_bits = np.array([-x.mean() for x in lp])
        out["bits"][name] = {"all": bits(every, lp), "kept": bits(keep, lp), "dropped": bits(drop, lp),
                             "kept_ci95_family_bootstrap": family_ci(per_bits[keep], lens[keep], fam[keep])}
    one = scores["1-gram"]
    for name in scores:
        for part, m in (("all", every), ("kept", keep), ("dropped", drop)):
            out["bits"][name][f"gain_over_1gram_{part}"] = bits(m, one) - bits(m, scores[name])
    return out


def run_score() -> dict:
    from .evaluate import protein_train_stream
    from .train import device

    d = load_protein("test_novel")
    idx = eval_indices("test_novel")
    seqs = [d["tokens"][d["offsets"][i] + 1: d["offsets"][i] + 1 + d["lengths"][i]].astype(np.int64) - 3 for i in idx]
    fam, lens = d["family"][idx], np.array([len(s) for s in seqs])

    scores = {}
    flat, tg = protein_train_stream()
    kn = ngram.KneserNey.fit(flat, tg, max(ORDERS), 21, 20)
    del flat, tg
    for o in ORDERS:
        scores[f"{o}-gram"] = ngram_logp(kn.table(o), o, seqs)
    del kn
    dev = device()
    for name in ("gpt", "lstm"):
        model = load_checkpoint(ROOT / "checkpoints" / f"protein_{name}.pt", dev)
        scores[name] = neural_logp(model, seqs, dev)

    out = {"eval_subset": {"sequences": int(len(idx)), "families": len(set(fam.tolist())),
                           "residues": int(lens.sum())}, "filters": {}}
    accs = (PROC / "protein_test_novel_accessions.txt").read_text().split()
    (RES / "homology").mkdir(parents=True, exist_ok=True)
    shared = novel_sharing_train(read_tsv(MM / "c30_cluster.tsv"))
    meta = json.loads((MM / "c30_meta.json").read_text())
    out["filters"]["cluster"] = {
        **meta, "min_seq_id": MIN_ID, "coverage": COV,
        "test_novel_sharing_cluster_with_train": len(shared), "test_novel_total": int(len(d["offsets"])),
        **filter_report(np.array([i in shared for i in idx]), scores, lens, fam)}
    (RES / "homology" / "test_novel_sharing_train_cluster30.txt").write_text(
        "".join(f"{accs[i]}\n" for i in sorted(shared)))
    if (MM / "s30_hits.m8").exists():
        hit = novel_with_hits(read_tsv(MM / "s30_hits.m8"))
        meta = json.loads((MM / "s30_meta.json").read_text())
        both = shared | hit
        out["filters"]["search"] = {**meta, "min_seq_id": SEARCH_ID, "query_coverage": SEARCH_QCOV,
                                    **filter_report(np.array([i in hit for i in idx]), scores, lens, fam)}
        out["filters"]["cluster_or_search"] = filter_report(np.array([i in both for i in idx]), scores, lens, fam)
        (RES / "homology" / "eval_subset_train_hit_search30.txt").write_text(
            "".join(f"{accs[i]}\n" for i in sorted(hit)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["cluster", "search", "score"])
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    if args.part in ("cluster", "search"):
        fn = run_cluster if args.part == "cluster" else run_search
        print(json.dumps(fn(args.threads), indent=2))
        return
    res = run_score()
    (RES / "homology_protein.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
