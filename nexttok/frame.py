"""Is the neural models' DNA advantage reading-frame inference?

GeneMark (Borodovsky & McIninch 1993) models coding DNA as an inhomogeneous,
3-periodic Markov chain: the distribution of a base given the five before it
depends on the base's codon position. Here every genomic class -- non-coding,
+ strand codon position 1-3, - strand codon position 1-3, and the few bases in
two overlapping genes -- gets its own Kneser-Ney smoothed table
P(x_i | x_{i-m..i-1}, class of x_i), fitted on the training blocks read in both
directions, exactly the stream the n-grams are fitted on.

The model is scored on the held-out targets of nexttok.evaluate in three ways:

  oracle   the target's class is read from the annotation (true frame and strand)
  window   posterior over the 7 classes (6 frames + non-coding) given the
           preceding k bases, assuming the whole window is in one class
  hmm      forward filtering in an HMM over the 8 classes, with transitions
           (gene starts and ends, the phase cycle) counted on the training
           labels; it sees exactly the context windows the neural models see

The LSTM, GPT and Kneser-Ney ladder are rescored on the same targets so every
genome gets the same table. E. coli uses the first-release 800-step
checkpoints in checkpoints/; other genomes use runs/<genome>/ (nexttok.genome).

  python -m nexttok.frame all --genome ecoli     # writes results/frame/ecoli.json
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from . import ngram
from .data import segments
from .evaluate import DNA_CTX, DNA_ORDERS, SWEEP_K, SWEEP_N, dna_neural_scores, dna_targets, symbol_logprobs
from .genome import GENOMES, checkpoint, use_genome
from .models import Config, build
from .train import device
from .vocab import N_SPECIAL

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "frame"
CACHE = ROOT / "results" / "cache"

N_CLASS = 8            # DNA_LABELS: non-coding, +1, +2, +3, -1, -2, -3, overlap
N_HYP = 7              # frame hypotheses: everything but "overlap"
CLASS_NAMES = ["non-coding", "+ codon 1", "+ codon 2", "+ codon 3", "− codon 1", "− codon 2", "− codon 3"]
ORDER = 5              # GeneMark's Markov order
MAX_ORDER = 7          # the oracle ladder goes up to the 8-gram's context
RC_CLASS = np.array([0, 4, 5, 6, 1, 2, 3, 7])  # + codon position c <-> - codon position c


# ----------------------------------------------------------------- bookkeeping
def reverse_complement(sym: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reverse complement of a labelled stretch (bases 0-3 = ACGT).

    A + strand gene read backwards on the other strand is a - strand gene with
    the same codon positions, so + codon c and - codon c swap.
    """
    return 3 - sym[::-1], RC_CLASS[labels[::-1]]


def window_classes(k: int) -> np.ndarray:
    """Class of each of the k context bases under each hypothesis about the target.

    Row h is the hypothesis "the target is in class h" (0 = non-coding,
    1-3 = + strand codon position, 4-6 = - strand codon position); column t is
    the t-th base of the window, so column k-1 is the base just before the
    target. Reading the + strand 5'->3', + strand codon positions run
    1, 2, 3, 1, ... and - strand ones 3, 2, 1, 3, ...
    """
    back = k - np.arange(k)  # steps from each context base to the target
    rows = [np.zeros(k, np.int64)]
    rows += [(h - 1 - back) % 3 + 1 for h in (1, 2, 3)]
    rows += [(h - 4 + back) % 3 + 4 for h in (4, 5, 6)]
    return np.stack(rows)


def train_stream(d: dict, max_ctx: int):
    """evaluate.dna_train_stream with the class of every target.

    Training blocks forwards and reverse-complemented; targets start max_ctx
    bases into each part. Returns (flat, targets, classes of the targets).
    """
    sym = d["tokens"].astype(np.int64) - N_SPECIAL
    parts, labs, tg, off = [], [], [], 0
    for a, b in segments(d["split"] == 0):
        for s, l in ((sym[a:b], d["labels"][a:b]), reverse_complement(sym[a:b], d["labels"][a:b])):
            parts.append(s)
            labs.append(l)
            tg.append(off + np.arange(max_ctx, len(s)))
            off += len(s)
    flat, lab, tg = np.concatenate(parts), np.concatenate(labs).astype(np.int64), np.concatenate(tg)
    return flat, tg, lab[tg]


# ----------------------------------------------------------------- model
class FrameMarkov:
    """Per-class Kneser-Ney Markov tables, class prior and class transitions."""

    def __init__(self, tables: dict[int, np.ndarray], prior: np.ndarray, trans: np.ndarray):
        self.tables = tables  # context length m -> (N_CLASS, 4**m, 4), rows sum to 1
        self.prior = prior    # (N_CLASS,) class frequency among training targets
        self.trans = trans    # (N_CLASS, N_CLASS): P(class of next base | class of this base)

    @classmethod
    def fit(cls, flat: np.ndarray, targets: np.ndarray, classes: np.ndarray,
            max_order: int = MAX_ORDER, pseudo: float = 0.5) -> "FrameMarkov":
        """classes[j] is the class of flat[targets[j]]; each target needs max_order bases before it."""
        per = [ngram.KneserNey.fit(flat, targets[classes == c], max_order + 1, 4, 4) for c in range(N_CLASS)]
        tables = {m: np.stack([kn.table(m + 1) for kn in per]) for m in range(max_order + 1)}
        prior = np.bincount(classes, minlength=N_CLASS) / len(classes)
        nxt = np.flatnonzero(np.diff(targets) == 1)  # consecutive bases of one part
        counts = np.zeros((N_CLASS, N_CLASS))
        np.add.at(counts, (classes[nxt], classes[nxt + 1]), 1)
        trans = (counts + pseudo) / (counts + pseudo).sum(1, keepdims=True)
        return cls(tables, prior, trans)

    # ------------------------------------------------------------- oracle
    def oracle(self, flat: np.ndarray, pos: np.ndarray, classes: np.ndarray, m: int):
        """log2 P(x | m preceding bases, true class) and the argmax prediction."""
        idx = ngram.context_index(flat, pos, m, 4) if m else np.zeros(len(pos), np.int64)
        rows = self.tables[m][classes, idx]
        return np.log2(rows[np.arange(len(pos)), flat[pos]]), rows.argmax(1)

    # ------------------------------------------------------------- window posterior
    def window(self, W: np.ndarray, order: int = ORDER):
        """Predict the last column of W from the k = W.shape[1] - 1 before it.

        Each context base is scored with the bases before it inside the window
        (up to `order`). Returns log2 P(target), the argmax prediction and the
        posterior over the 7 frame hypotheses, shape (n, 7).
        """
        n, k = W.shape[0], W.shape[1] - 1
        E = np.empty((n, k, N_CLASS))
        idx = np.zeros(n, np.int64)
        for t in range(k):
            E[:, t] = np.log2(self.tables[min(t, order)][:, idx, W[:, t]]).T
            idx = (idx * 4 + W[:, t]) % 4 ** order
        loglik = E[:, np.arange(k), window_classes(k)].sum(-1) if k else np.zeros((n, N_HYP))
        logpost = loglik + np.log2(self.prior[:N_HYP] / self.prior[:N_HYP].sum())
        post = np.exp2(logpost - logpost.max(1, keepdims=True))
        post /= post.sum(1, keepdims=True)
        dist = np.einsum("nh,hnv->nv", post, self.tables[min(k, order)][:N_HYP, idx])
        return np.log2(dist[np.arange(n), W[:, k]]), dist.argmax(1), post

    # ------------------------------------------------------------- hmm
    def hmm(self, W: np.ndarray, start: int, order: int = ORDER):
        """Forward filtering along each row of W, scoring columns >= start.

        The class distribution starts at the training prior at column 0. Returns,
        for the scored columns, log2 P(x_t | x_<t in the row), the argmax
        prediction, and P(class of x_t | x_<t) of shape (n, L - start, N_CLASS).
        """
        n, L = W.shape
        state = np.tile(self.prior, (n, 1))
        idx = np.zeros(n, np.int64)
        lp, pred, cls = [], [], []
        for t in range(L):
            rows = self.tables[min(t, order)][:, idx]          # (N_CLASS, n, 4)
            joint = state * rows[:, np.arange(n), W[:, t]].T   # (n, N_CLASS)
            px = joint.sum(1)
            if t >= start:
                lp.append(np.log2(px))
                pred.append(np.einsum("nc,cnv->nv", state, rows).argmax(1))
                cls.append(state)
            state = (joint / px[:, None]) @ self.trans
            idx = (idx * 4 + W[:, t]) % 4 ** order
        return np.stack(lp, 1), np.stack(pred, 1), np.stack(cls, 1)


# ----------------------------------------------------------------- scoring helpers
def by_class(logp: np.ndarray, labels: np.ndarray) -> dict:
    out = {name: float(-logp[labels == c].mean()) for c, name in enumerate(CLASS_NAMES)}
    out["all"] = float(-logp.mean())
    return out


def class_accuracy(post: np.ndarray, labels: np.ndarray) -> dict:
    """Argmax accuracy of a class distribution over the 7 frame hypotheses (overlap bases excluded)."""
    keep = labels < N_HYP
    guess = post[:, :N_HYP].argmax(1)
    coding = keep & (labels > 0)
    return {"accuracy_7class": float((guess[keep] == labels[keep]).mean()),
            "frame_accuracy_on_coding": float((guess[coding] == labels[coding]).mean()), "n": int(keep.sum())}


def scoring_windows(sym: np.ndarray, segs, T: int = 512):
    """The windows dna_neural_scores uses: length T, stride T - DNA_CTX, tails scored.

    Yields (window matrix, positions of its scored columns) grouped by window length.
    """
    step = T - DNA_CTX
    groups: dict[int, list[int]] = {}
    for a, b in segs:
        for s in range(a, b - DNA_CTX, step):
            groups.setdefault(min(s + T, b) - s, []).append(s)
    for L, starts in groups.items():
        starts = np.array(starts)
        yield sym[starts[:, None] + np.arange(L)], (starts[:, None] + np.arange(DNA_CTX, L)).reshape(-1)


def preceding_windows(sym: np.ndarray, pos: np.ndarray, k: int, chunk: int = 5000):
    """(rows, k + 1) matrices: the k bases before each position, then the base itself."""
    for i in range(0, len(pos), chunk):
        p = pos[i: i + chunk]
        yield sym[p[:, None] + np.arange(-k, 1)]


def load_checkpoint(path: Path, dev):
    ck = torch.load(path, map_location="cpu")
    m = build(ck["model"], Config(**ck["cfg"]))
    m.load_state_dict(ck["state"])
    return m.to(dev).eval()


def sweep_positions(tg: np.ndarray) -> np.ndarray:
    """The targets of evaluate.sweep_targets (same seed, same draw), as genome positions."""
    return np.random.default_rng(7).choice(tg, size=SWEEP_N, replace=False)


# ----------------------------------------------------------------- experiment
# Three stages, so the CPU and GPU halves can run separately and each process
# stays small: `markov` (n-gram ladder and frame models), `neural` (LSTM, GPT),
# `summary` (results/frame/<genome>.json). Per-target arrays meet in
# results/cache/frame_<genome>.npz.
def targets(genome: str):
    """Test targets as nexttok.evaluate scores them, plus the 20,000 sweep rows."""
    with use_genome(genome):
        from .evaluate import load_dna
        d = load_dna()
        sym, tg, labels, segs = dna_targets()
    pick = sweep_positions(tg)
    rows = sym[pick[:, None] + np.arange(-max(SWEEP_K), 1)]  # 256 context bases, then the target
    return d, sym, tg, labels.astype(np.int64), segs, d["labels"][pick].astype(np.int64), rows


def cache_path(genome: str) -> Path:
    return CACHE / f"frame_{genome}.npz"


def save(genome: str, arrays: dict, meta: dict | None = None) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = cache_path(genome)
    old = dict(np.load(path)) if path.exists() else {}
    old.update({k: (v.astype(np.float32) if v.dtype == np.float64 else v) for k, v in arrays.items()})
    np.savez(path, **old)
    if meta:
        mp = path.with_suffix(".json")
        mp.write_text(json.dumps({**(json.loads(mp.read_text()) if mp.exists() else {}), **meta}))


def run_markov(genome: str) -> None:
    d, sym, tg, labels, segs, lab_s, rows = targets(genome)
    K = max(SWEEP_K)
    flat_s, end = rows.reshape(-1), np.arange(len(rows)) * rows.shape[1] + K
    with use_genome(genome):
        from .evaluate import dna_train_stream
        flat, tr = dna_train_stream(max(DNA_ORDERS) - 1)
    a = {"test/labels": labels, "sweep/labels": lab_s}

    # Kneser-Ney ladder on the stream and targets of evaluate.run_ngram. Fitting each
    # order on its own gives the same tables as KneserNey(N=12).table(o), with a
    # fraction of the memory; the 12-gram goes first, while little else is held.
    for o in sorted(DNA_ORDERS, reverse=True):
        table = ngram.KneserNey.fit(flat, tr, o, 4, 4).tables[o]
        logp, pred, _ = ngram.score(table, o, sym, tg, 4, 4)
        a[f"test/{o}-gram"], a[f"test/{o}-gram/correct"] = logp, pred == sym[tg]
        a[f"sweep/ngram/c{o - 1}"] = ngram.score(table, o, flat_s, end, 4, 4)[0]  # keyed by context length
        del table
        gc.collect()

    # frame-aware Markov chains on the identical training stream
    f2, tr2, cls = train_stream(d, max(DNA_ORDERS) - 1)
    assert np.array_equal(f2, flat) and np.array_equal(tr2, tr)
    del f2, tr2
    fm = FrameMarkov.fit(flat, tr, cls)
    del flat, tr, cls
    gc.collect()

    for m in range(MAX_ORDER + 1):
        logp, pred = fm.oracle(sym, tg, labels, m)
        a[f"test/frame_oracle_o{m}"], a[f"test/frame_oracle_o{m}/correct"] = logp, pred == sym[tg]

    lp, pr, post = (np.concatenate(x) for x in zip(*(fm.window(W) for W in preceding_windows(sym, tg, K))))
    a[f"test/frame_window_k{K}"], a[f"test/frame_window_k{K}/correct"] = lp, pr == sym[tg]
    a[f"test/frame_window_k{K}/post"] = post

    pos, parts = [], []
    for W, p in scoring_windows(sym, segs):
        x, y, z = fm.hmm(W, DNA_CTX)
        pos.append(p)
        parts.append((x.reshape(-1), y.reshape(-1), z.reshape(-1, N_CLASS)))
    pos = np.concatenate(pos)
    order = np.argsort(pos)
    assert np.array_equal(pos[order], tg)  # same targets as the neural models, same context each
    lp, pr, st = (np.concatenate(x)[order] for x in zip(*parts))
    a["test/frame_hmm"], a["test/frame_hmm/correct"], a["test/frame_hmm/state"] = lp, pr == sym[tg], st

    for k in SWEEP_K:
        a[f"sweep/frame_oracle/k{k}"] = fm.oracle(flat_s, end, lab_s, min(k, ORDER))[0]
        lp, _, post = (np.concatenate(x) for x in zip(*(fm.window(rows[i: i + 5000, K - k:])
                                                         for i in range(0, len(rows), 5000))))
        a[f"sweep/frame_window/k{k}"], a[f"sweep/frame_window/k{k}/post"] = lp, post
        lp, _, st = fm.hmm(rows[:, K - k:], k)
        a[f"sweep/frame_hmm/k{k}"], a[f"sweep/frame_hmm/k{k}/state"] = lp[:, 0], st[:, 0]
    save(genome, a, {"class_prior": fm.prior.tolist(), "transitions": fm.trans.tolist()})


def run_neural(genome: str) -> None:
    _, sym, tg, _, segs, _, rows = targets(genome)
    K = max(SWEEP_K)
    t = torch.from_numpy(rows[:, -1])
    dev = device()
    a = {}
    for name in ("lstm", "gpt"):
        model = load_checkpoint(checkpoint(genome, name), dev)
        logp, corr = dna_neural_scores(model, sym, segs, dev)
        a[f"test/{name}"], a[f"test/{name}/correct"] = logp, corr
        for k in SWEEP_K:
            lps = []
            for i in range(0, len(rows), 512):
                x = torch.from_numpy(rows[i: i + 512, K - k: K] + N_SPECIAL).to(dev)
                lps.append(symbol_logprobs(model, x, "dna")[:, -1].cpu())
            a[f"sweep/{name}/k{k}"] = torch.cat(lps).gather(1, t[:, None])[:, 0].numpy()
        del model
        if dev.type == "mps":
            torch.mps.empty_cache()
    save(genome, a)


def summary(genome: str) -> dict:
    c = np.load(cache_path(genome))
    meta = json.loads(cache_path(genome).with_suffix(".json").read_text())
    labels, lab_s = c["test/labels"], c["sweep/labels"]
    ladder = {f"{o}-gram": float(-c[f"test/{o}-gram"].mean()) for o in DNA_ORDERS}
    best = min(ladder, key=ladder.get)  # chosen on test, as in the README
    K = max(SWEEP_K)
    names = [best] + [f"frame_oracle_o{m}" for m in range(MAX_ORDER + 1)] + [f"frame_window_k{K}", "frame_hmm", "lstm", "gpt"]
    test = {n: {**by_class(c[f"test/{n}"], labels), "top1": float(c[f"test/{n}/correct"].mean())}
            for n in names if f"test/{n}" in c}
    last = (lab_s == 3) | (lab_s == 4)  # + codon 3 and - codon 1: the last base read in each codon
    sweep = {"k": SWEEP_K, "n": int(len(lab_s)), "n_codon_read_last": int(last.sum()), "codon_read_last": {}}
    for n in ("lstm", "gpt", "frame_oracle", "frame_window", "frame_hmm"):
        if f"sweep/{n}/k{K}" in c:
            sweep[n] = {k: float(-c[f"sweep/{n}/k{k}"].mean()) for k in SWEEP_K}
            sweep["codon_read_last"][n] = {k: float(-c[f"sweep/{n}/k{k}"][last].mean()) for k in SWEEP_K}
    ctx = [o - 1 for o in DNA_ORDERS]
    sweep["ngram"] = {m: float(-c[f"sweep/ngram/c{m}"].mean()) for m in ctx}
    sweep["codon_read_last"]["ngram"] = {m: float(-c[f"sweep/ngram/c{m}"][last].mean()) for m in ctx}
    return {
        "genome": genome, "species": GENOMES[genome].species, "order": ORDER, "n_test": int(len(labels)),
        "ngram_ladder": ladder, "best_ngram": best, **meta, "test": test,
        "class_inference": {f"frame_window_k{K}": class_accuracy(c[f"test/frame_window_k{K}/post"], labels),
                            "frame_hmm": class_accuracy(c["test/frame_hmm/state"], labels)},
        "sweep": sweep,
        "sweep_class_inference": {
            "frame_window": {k: class_accuracy(c[f"sweep/frame_window/k{k}/post"], lab_s) for k in SWEEP_K},
            "frame_hmm": {k: class_accuracy(c[f"sweep/frame_hmm/k{k}/state"], lab_s) for k in SWEEP_K}},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["markov", "neural", "summary", "all"])
    ap.add_argument("--genome", choices=sorted(GENOMES), default="ecoli")
    args = ap.parse_args()
    if args.stage in ("markov", "all"):
        run_markov(args.genome)
    if args.stage in ("neural", "all"):
        run_neural(args.genome)
    if args.stage in ("summary", "all"):
        res = summary(args.genome)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"{args.genome}.json").write_text(json.dumps(res, indent=2))
        print(json.dumps({k: res[k] for k in ("best_ngram", "test", "class_inference")}, indent=1)[:6000])


if __name__ == "__main__":
    main()
