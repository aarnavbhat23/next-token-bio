"""Whole-sequence scoring shared by the homology filter and the variant-effect study.

Every score is log2 P(residue) renormalised over the 20 amino acids, exactly as
in `evaluate.run_neural`: the model reads BOS + seq[:-1] and predicts seq.
Checkpoints are loaded by path so that any `checkpoints/protein_*.pt` can be
re-scored.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from . import ngram
from .models import Config, build
from .vocab import AMINO, BOS, N_SPECIAL, symbol_ids

ROOT = Path(__file__).resolve().parents[1]
PAD = 20  # n-gram context symbol before the first residue (as in evaluate.protein_targets)

_LUT = np.frombuffer(b"\0" * N_SPECIAL + AMINO.encode(), dtype=np.uint8)


def to_str(tokens: np.ndarray) -> str:
    """Residue token ids (3..22) to a one-letter string, vectorised."""
    return _LUT[tokens].tobytes().decode()


def to_ids(seq: str) -> np.ndarray:
    """One-letter string to residue ids 0..19 (raises on non-standard residues)."""
    idx = np.array([AMINO.index(c) for c in seq], dtype=np.int64)
    return idx


def load_checkpoint(path: str | Path, dev) -> torch.nn.Module:
    ck = torch.load(path, map_location="cpu")
    m = build(ck["model"], Config(**ck["cfg"]))
    m.load_state_dict(ck["state"])
    return m.to(dev).eval()


@torch.no_grad()
def neural_logp(model, seqs: list[np.ndarray], dev, batch: int = 32) -> list[np.ndarray]:
    """Per-residue log2 p for each sequence of residue ids 0..19."""
    order = np.argsort([len(s) for s in seqs])
    out: list[np.ndarray | None] = [None] * len(seqs)
    sym = symbol_ids("protein")
    for i in range(0, len(order), batch):
        idx = order[i: i + batch]
        L = max(len(seqs[j]) for j in idx)
        x = torch.zeros(len(idx), L, dtype=torch.long)
        for r, j in enumerate(idx):
            s = torch.from_numpy(np.asarray(seqs[j], np.int64) + N_SPECIAL)
            x[r, 0] = BOS
            x[r, 1: len(s)] = s[:-1]
        logits = model(x.to(dev))[..., sym].float()
        lp = (torch.log_softmax(logits, -1) / np.log(2)).cpu()
        for r, j in enumerate(idx):
            t = torch.from_numpy(np.asarray(seqs[j], np.int64))
            out[j] = lp[r, : len(t)].gather(1, t[:, None])[:, 0].numpy()
    return out


def ngram_logp(table: np.ndarray, order: int, seqs: list[np.ndarray]) -> list[np.ndarray]:
    """Per-residue log2 p under an n-gram table, BOS-padded like evaluate.protein_targets."""
    pad = max(order - 1, 0)
    flat = np.concatenate([np.concatenate([np.full(pad, PAD), np.asarray(s, np.int64)]) for s in seqs])
    lens = np.array([len(s) for s in seqs])
    starts = np.concatenate([[0], np.cumsum(lens + pad)[:-1]]) + pad
    tg = np.concatenate([st + np.arange(n) for st, n in zip(starts, lens)])
    logp, _, _ = ngram.score(table, order, flat, tg, 21, 20)
    return np.split(logp, np.cumsum(lens)[:-1])
