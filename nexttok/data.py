"""Loading processed splits and sampling training windows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .vocab import BOS, N_SPECIAL

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
SPLIT_SEED = 0  # which family split load_protein reads; set once per process


def set_split_seed(seed: int) -> None:
    global SPLIT_SEED
    SPLIT_SEED = seed


def protein_dir(split_seed: int | None = None) -> Path:
    s = SPLIT_SEED if split_seed is None else split_seed
    d = PROC / f"split{s}"
    if s == 0 and not d.exists() and (PROC / "protein_train.npz").exists():
        return PROC  # checkout prepared before per-split folders: split 0 lived here
    return d


def load_protein(split: str) -> dict[str, np.ndarray]:
    return dict(np.load(protein_dir() / f"protein_{split}.npz"))


def load_dna() -> dict[str, np.ndarray]:
    return dict(np.load(PROC / "dna_ecoli.npz"))


def protein_subset(split: str, n: int, seed: int = 0, families: bool = False):
    """A fixed random subset of sequences (residue token ids, no BOS/EOS).

    With families=True, also return each picked sequence's family id.
    """
    d = load_protein(split)
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(len(d["offsets"]), size=min(n, len(d["offsets"])), replace=False))
    seqs = [d["tokens"][d["offsets"][i] + 1: d["offsets"][i] + 1 + d["lengths"][i]] for i in pick]
    return (seqs, d["family"][pick]) if families else seqs


def segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous [start, end) runs where mask is True."""
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


class ProteinSampler:
    """Windows of the packed BOS/seq/EOS stream.

    Half of the windows start at a BOS, half at a random offset, so the model
    also learns to predict from mid-sequence contexts that lack a BOS (needed
    for the truncated-context experiment).
    """

    def __init__(self, split: str, T: int, seed: int):
        d = load_protein(split)
        self.tokens = torch.from_numpy(d["tokens"].astype(np.int64))
        self.offsets = d["offsets"]
        self.T = T
        self.rng = np.random.default_rng(seed)

    def batch(self, B: int):
        n = len(self.tokens) - self.T - 1
        bos = self.offsets[self.rng.integers(0, len(self.offsets), B)]
        rnd = self.rng.integers(0, n, B)
        starts = np.minimum(np.where(self.rng.random(B) < 0.5, bos, rnd), n)
        x = torch.stack([self.tokens[s: s + self.T + 1] for s in starts])
        return x[:, :-1], x[:, 1:]


class DNASampler:
    """Random windows inside training blocks, reverse-complemented half the time."""

    def __init__(self, split_id: int, T: int, seed: int):
        d = load_dna()
        self.tokens = torch.from_numpy(d["tokens"].astype(np.int64))
        self.segs = [(a, b) for a, b in segments(d["split"] == split_id) if b - a > T + 1]
        w = np.array([b - a - T - 1 for a, b in self.segs], dtype=np.float64)
        self.p = w / w.sum()
        self.T = T
        self.rng = np.random.default_rng(seed)

    def batch(self, B: int):
        seg = self.rng.choice(len(self.segs), size=B, p=self.p)
        out = []
        for s in seg:
            a, b = self.segs[s]
            st = self.rng.integers(a, b - self.T - 1)
            w = self.tokens[st: st + self.T + 1]
            if self.rng.random() < 0.5:  # reverse complement: A<->T (3<->6), C<->G (4<->5)
                w = (2 * N_SPECIAL + 3) - w.flip(0)
            out.append(w)
        x = torch.stack(out)
        return x[:, :-1], x[:, 1:]


def with_bos(seq: np.ndarray) -> np.ndarray:
    return np.concatenate([[BOS], seq]).astype(np.int64)
