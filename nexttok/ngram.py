"""Interpolated Kneser-Ney n-gram models, fully dense and vectorised.

Contexts are encoded as base-C integers with the *leftmost* symbol as the most
significant digit, so dropping the oldest symbol is a reshape+sum over axis 0.
Every probability table is a dense array of shape (C**(n-1), V), which is
cheap for the alphabets used here (protein: C=21, n<=6; DNA: C=4, n<=12).
"""

from __future__ import annotations

import numpy as np


def context_index(flat: np.ndarray, pos: np.ndarray, k: int, C: int) -> np.ndarray:
    """Index of the k symbols immediately before each position in `pos`."""
    idx = np.zeros(len(pos), dtype=np.int64)
    for j in range(k, 0, -1):  # oldest first -> most significant
        idx = idx * C + flat[pos - j]
    return idx


def count_top(flat: np.ndarray, targets: np.ndarray, N: int, C: int, V: int) -> np.ndarray:
    """Raw counts of order-N events; shape (C**(N-1), V)."""
    ctx = context_index(flat, targets, N - 1, C)
    ev = ctx * V + flat[targets]
    return np.bincount(ev, minlength=C ** (N - 1) * V).reshape(C ** (N - 1), V)


def _discounts(counts: np.ndarray) -> tuple[float, float, float]:
    """Modified Kneser-Ney discounts D1, D2, D3+ (Chen & Goodman, 1998)."""
    n = [int((counts == i).sum()) for i in range(1, 5)]
    if min(n) == 0:  # dense table, nothing to estimate from: mild fixed discounts
        return 0.5, 1.0, 1.5
    Y = n[0] / (n[0] + 2 * n[1])
    raw = (1 - 2 * Y * n[1] / n[0], 2 - 3 * Y * n[2] / n[1], 3 - 4 * Y * n[3] / n[2])
    # the estimates can leave [0, c] on tables with few distinct counts
    return tuple(float(np.clip(d, 0.05, c - 0.05)) for d, c in zip(raw, (1, 2, 3)))


def _interp(counts: np.ndarray, lower: np.ndarray, C: int) -> np.ndarray:
    """P(w|h) for one level given counts at this level and P(w|h') below."""
    D1, D2, D3 = _discounts(counts)
    D = np.select([counts == 1, counts == 2, counts >= 3], [D1, D2, D3], 0.0)
    counts = counts.astype(np.float64)
    tot = counts.sum(1, keepdims=True)
    # lower has shape (C**(k-1), V); broadcast across the dropped leftmost symbol
    low = np.broadcast_to(lower[None], (C,) + lower.shape).reshape(counts.shape)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = (counts - D) / tot + (D.sum(1, keepdims=True) / tot) * low
    return np.where(tot > 0, p, low)


class KneserNey:
    def __init__(self, top_counts: np.ndarray, N: int, C: int, V: int):
        self.N, self.C, self.V = N, C, V
        # raw counts for every order, derived by marginalising the oldest symbol
        raw = {N: top_counts}
        for n in range(N - 1, 0, -1):
            raw[n] = raw[n + 1].reshape(C, -1, V).sum(0)
        self.tables = {}
        # unigram level: continuation counts interpolated with uniform
        cont1 = (raw[2] > 0).sum(0) if N >= 2 else raw[1][0]
        p1 = (cont1 + 1.0) / (cont1.sum() + V)
        self.tables[1] = p1[None, :]
        for n in range(2, N + 1):
            # lower levels use continuation counts, the top level raw counts
            if n < N:
                cnt = (raw[n + 1].reshape(C, -1, V) > 0).sum(0)
            else:
                cnt = raw[n]
            self.tables[n] = _interp(cnt, self.tables[n - 1], C)

    def table(self, order: int) -> np.ndarray:
        """Highest-order table for an `order`-gram model.

        A lower-order *model* uses raw counts at its own top level, so it is
        rebuilt rather than read off the continuation tables used inside a
        higher-order model.
        """
        if order == self.N:
            return self.tables[order]
        sub = KneserNey(self._raw_top(order), order, self.C, self.V)
        return sub.tables[order]

    def _raw_top(self, order: int) -> np.ndarray:
        r = self._top
        for _ in range(self.N - order):
            r = r.reshape(self.C, -1, self.V).sum(0)
        return r

    @classmethod
    def fit(cls, flat, targets, N, C, V):
        top = count_top(flat, targets, N, C, V)
        m = cls(top, N, C, V)
        m._top = top
        return m


def score(table: np.ndarray, order: int, flat: np.ndarray, targets: np.ndarray, C: int, V: int):
    """Log2-probabilities and argmax predictions for each target position."""
    ctx = context_index(flat, targets, order - 1, C) if order > 1 else np.zeros(len(targets), np.int64)
    rows = table[ctx]
    logp = np.log2(rows[np.arange(len(targets)), flat[targets]])
    return logp, rows.argmax(1), rows
