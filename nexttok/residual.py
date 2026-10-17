"""What is the LSTM's ~0.03-bit residual beyond per-protein composition?

At k = 256 residues of context on held-out families, the 3-gram + own-composition
cache of `evaluate.run_cache` recovers about half of the LSTM's gain over the
3-gram. This module adds candidate explanations as explicit baselines and scores
them on exactly the 20,000 test targets of the context sweep. Feature weights are
estimated on targets from *training* families, as the n-gram itself is; mixing
weights and L2 strength are chosen on *validation* families, as run_cache does.
(Fitting the feature weights on validation families instead overfits to those
few families and hurts held-out loss; that variant is reported as a warning.)

  periodic   hydrophobic / not at the residues 2, 3, 4 and 7 behind the target
             (strand faces repeat every 2 residues, helix faces every 3.6), all
             16 joint patterns, as a log-linear correction to the cache model.
             The same features linearly mixed in, as cache_protein does with
             composition, are reported too.
  local      composition of the last 8, 16, ..., 128 residues as well as the
             last 256: a protein's composition changes along its length.
  lags       full residue identity at lags 2-16, additively: every pairwise
             lag statistic, periodic or not.

The per-target gain of the LSTM over the cache model is then split by
predicted disorder (metapredict, if installed) and by a hydrophobic-moment
proxy for helix- or strand-like periodicity.

  python -m nexttok.residual                 # writes results/residual_protein.json
  python -m nexttok.residual --splits-only   # redo only the per-target splits
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import ngram
from .data import load_protein, protein_subset
from .seqscore import ROOT, load_checkpoint, to_str
from .vocab import AMINO, N_SPECIAL, symbol_ids

RES = ROOT / "results"
K = 256                                     # context length of the comparison
N_EVAL, SWEEP_N, SWEEP_SEED = 3000, 20000, 7  # as evaluate.sweep_targets
TRAIN_N, VAL_N, FIT_SEED = 500_000, 200_000, 11  # fitting targets (training / validation families)
HYDROPHOBIC = "AVILMFWC"
CLASSES3 = ("AVILMFWC", "GHNPQSTY", "DEKR")  # hydrophobic / polar / charged
PERIODIC_LAGS = (2, 3, 4, 7)
WINDOWS = (8, 16, 32, 64, 128, 256)
ALPHAS = (0.5, 2.0, 8.0, 32.0)
ID_LAGS = tuple(range(2, 17))
SCAN_LAGS = tuple(range(1, 17))
L2_GRID = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
LN2 = math.log(2)
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)

# Eisenberg consensus hydrophobicity and Kyte-Doolittle hydropathy, in AMINO order
EISENBERG = dict(A=0.62, C=0.29, D=-0.90, E=-0.74, F=1.19, G=0.48, H=-0.40, I=1.38, K=-1.50, L=1.06,
                 M=0.64, N=-0.78, P=0.12, Q=-0.85, R=-2.53, S=-0.18, T=-0.05, V=1.08, W=0.81, Y=0.26)
KYTE_DOOLITTLE = dict(A=1.8, C=2.5, D=-3.5, E=-3.5, F=2.8, G=-0.4, H=-3.2, I=4.5, K=-3.9, L=3.8,
                      M=1.9, N=-3.5, P=-1.6, Q=-3.5, R=-4.5, S=-0.8, T=-0.7, V=4.2, W=-0.9, Y=-1.3)


# ------------------------------------------------------------------ features
def class_map(groups) -> np.ndarray:
    """Residue id (0..19) -> class id. Residues not listed form the last class."""
    m = np.full(20, len(groups), np.int64)
    for c, g in enumerate(groups):
        for a in g:
            m[AMINO.index(a)] = c
    return m


HYDRO2 = class_map([HYDROPHOBIC])
HYDRO3 = class_map(CLASSES3[:2])  # charged = the remainder


def lag_pattern(ctx: np.ndarray, lags, cmap: np.ndarray) -> tuple[np.ndarray, int]:
    """Joint class pattern of the residues `lags` behind each target (ctx[:, -1] is lag 1)."""
    n_cls = int(cmap.max()) + 1
    idx = np.zeros(len(ctx), np.int64)
    for L in lags:
        idx = idx * n_cls + cmap[ctx[:, -L]]
    return idx, n_cls ** len(lags)


def window_counts(ctx: np.ndarray, windows) -> dict[int, np.ndarray]:
    """Residue counts in the last w positions of each context, for every w."""
    out = {w: np.zeros((len(ctx), 20), np.float32) for w in windows}
    rev = ctx[:, ::-1]
    for a in range(20):
        cs = np.cumsum(rev == a, axis=1, dtype=np.int16)
        for w in windows:
            out[w][:, a] = cs[:, w - 1]
    return out


def hydrophobic_moment(h: np.ndarray, deg: float, j: np.ndarray | None = None) -> float:
    """Eisenberg hydrophobic moment per residue at `deg` degrees per residue; `j` = positions."""
    j = (np.arange(len(h)) if j is None else np.asarray(j)) * math.radians(deg)
    return float(np.hypot((h * np.cos(j)).sum(), (h * np.sin(j)).sum()) / len(h))


# ------------------------------------------------------------------ targets
def long_context_targets(seqs: list[np.ndarray], n: int, seed: int, k: int = K):
    """Random targets with >= k residues of context, drawn exactly as evaluate.sweep_targets.

    `seqs` hold residue ids 0..19. Returns ctx (N, k) uint8, tgt (N,), seq index, position.
    """
    lens = np.array([len(s) for s in seqs])
    sid = np.repeat(np.arange(len(seqs)), np.clip(lens - k, 0, None))
    pos = np.concatenate([np.arange(k, L) for L in lens] + [np.zeros(0, np.int64)])
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(sid), size=min(n, len(sid)), replace=False)
    sid, pos = sid[pick], pos[pick]
    ctx = np.stack([seqs[i][p - k: p] for i, p in zip(sid, pos)]).astype(np.uint8)
    tgt = np.array([seqs[i][p] for i, p in zip(sid, pos)], np.int64)
    return ctx, tgt, sid, pos


def packed_targets(d: dict, n: int, seed: int, k: int = K):
    """`n` random targets with >= k residues of context from a packed split (load_protein)."""
    lens = d["lengths"]
    m = np.clip(lens - k, 0, None)
    sid = np.repeat(np.arange(len(lens)), m)
    pos = k + np.arange(int(m.sum())) - np.repeat(np.cumsum(m) - m, m)
    pick = np.random.default_rng(seed).choice(len(sid), size=min(n, len(sid)), replace=False)
    at = d["offsets"][sid[pick]] + 1 + pos[pick]  # token index of each target (offset + BOS)
    tok = d["tokens"]
    ctx = (tok[at[:, None] - k + np.arange(k)] - N_SPECIAL).astype(np.uint8)
    return ctx, tok[at].astype(np.int64) - N_SPECIAL


def residues(tokens) -> np.ndarray:
    return np.asarray(tokens, np.int64) - N_SPECIAL


# ------------------------------------------------------------------ fitting
def fit_mixture(P: np.ndarray, iters: int = 500, tol: float = 1e-9) -> np.ndarray:
    """Maximum-likelihood weights of a linear mixture by EM.

    P[n, c] is the probability component c gives the true target of example n.
    """
    w = np.full(P.shape[1], 1.0 / P.shape[1])
    prev = -np.inf
    for _ in range(iters):
        mix = P @ w
        ll = float(np.log(mix).mean())
        w = (P * w / mix[:, None]).mean(0)
        if ll - prev < tol:
            break
        prev = ll
    return w


def fit_loglinear(log_base: np.ndarray, feats: list[tuple[np.ndarray, int]], tgt: np.ndarray, l2: float):
    """Weights W_f (n_f, 20) maximising mean log p, p ∝ exp(log_base + Σ_f W_f[idx_f]).

    Convex; solved with L-BFGS. `log_base` is natural-log probabilities.
    """
    lb = torch.from_numpy(np.ascontiguousarray(log_base, np.float64))
    t = torch.from_numpy(tgt)
    idx = [torch.from_numpy(i) for i, _ in feats]
    W = [torch.zeros(n, 20, dtype=torch.float64, requires_grad=True) for _, n in feats]
    opt = torch.optim.LBFGS(W, lr=1, max_iter=500, tolerance_grad=1e-9, tolerance_change=1e-12,
                            history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = lb + sum(w[i] for w, i in zip(W, idx))
        loss = F.cross_entropy(logits, t) + l2 * sum((w ** 2).sum() for w in W)
        loss.backward()
        return loss

    opt.step(closure)
    return [w.detach().numpy() for w in W]


def apply_loglinear(log_base, feats, W) -> np.ndarray:
    """Log2 probability of each target is taken by the caller; this returns natural-log rows."""
    z = log_base + sum(w[i] for w, (i, _) in zip(W, feats))
    return z - np.logaddexp.reduce(z, axis=1, keepdims=True)


def target_bits(log_rows: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Per-target log2 p from natural-log rows."""
    return log_rows[np.arange(len(tgt)), tgt] / LN2


# ------------------------------------------------------------------ one target set
class Parts:
    """The ingredients every baseline needs, computed once per target set."""

    def __init__(self, ctx, tgt, table3, bg):
        self.ctx, self.tgt, self.bg = ctx, tgt, bg
        self.p3 = table3[ctx[:, -2].astype(np.int64) * 21 + ctx[:, -1]]
        self.counts = window_counts(ctx, WINDOWS)

    def cache(self, w: int, a: float) -> np.ndarray:
        return (self.counts[w] + a * self.bg) / (w + a)

    def at(self, rows: np.ndarray) -> np.ndarray:
        return rows[np.arange(len(self.tgt)), self.tgt]

    def components(self):
        """(name, rows) for the local-composition mixture: 3-gram + caches."""
        yield "3-gram", self.p3
        for w in WINDOWS:
            for a in ALPHAS:
                yield f"cache{w}_a{a:g}", self.cache(w, a)

    def mix(self, weights) -> np.ndarray:
        out = np.zeros_like(self.p3)
        for wt, (_, rows) in zip(weights, self.components()):
            out += wt * rows
        return out


def grid_cache(parts: Parts):
    """evaluate.run_cache's grid search at k = K (same grid, same objective)."""
    lams = np.linspace(0, 1, 21)
    alphas = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
    p_ng, hits, b = parts.at(parts.p3), parts.counts[K][np.arange(len(parts.tgt)), parts.tgt], parts.bg[parts.tgt]
    grid = [(float(-np.log2(l * p_ng + (1 - l) * (hits + a * b) / (K + a)).mean()), float(l), float(a))
            for l in lams for a in alphas]
    _, l, a = min(grid)
    return l, a


# ------------------------------------------------------------------ neural
@torch.no_grad()
def neural_target_bits(path, ctx: np.ndarray, tgt: np.ndarray, dev, batch: int = 512) -> np.ndarray:
    """log2 p(target | last K residues), no BOS, as evaluate.run_context."""
    model = load_checkpoint(path, dev)
    out = []
    for i in range(0, len(ctx), batch):
        x = torch.from_numpy(ctx[i: i + batch].astype(np.int64) + N_SPECIAL).to(dev)
        lp = torch.log_softmax(model(x)[:, -1][:, symbol_ids("protein")].float(), -1) / LN2
        out.append(lp.cpu().gather(1, torch.from_numpy(tgt[i: i + batch])[:, None])[:, 0].numpy())
    return np.concatenate(out)


# ------------------------------------------------------------------ per-target labels
def structure_labels(seqs, sid, pos, half: int = 5) -> dict[str, np.ndarray]:
    """Hydrophobic-moment proxy labels for each target, from the true sequence around it.

    helix-like / strand-like: the 11-residue window centred on the target has a
    hydrophobic moment at 100 deg (alpha helix) or 170 deg (beta strand) in the top
    third of all targets, and larger (relative to that threshold) than the other.
    Also: mean Kyte-Doolittle over 19 residues >= 1.6 (transmembrane-like).
    The target residue itself is left out of every window, so no label encodes the
    identity being predicted.
    """
    eis = np.array([EISENBERG[a] for a in AMINO])
    kd = np.array([KYTE_DOOLITTLE[a] for a in AMINO])
    m100, m170, hyd = [], [], []
    for i, p in zip(sid, pos):
        s = seqs[i]
        j = np.arange(max(0, p - half), min(len(s), p + half + 1))
        j = j[j != p]
        m100.append(hydrophobic_moment(eis[s[j]], 100, j))
        m170.append(hydrophobic_moment(eis[s[j]], 170, j))
        w = np.arange(max(0, p - 9), min(len(s), p + 10))
        hyd.append(kd[s[w[w != p]]].mean())
    m100, m170, hyd = map(np.array, (m100, m170, hyd))
    t100, t170 = np.quantile(m100, 2 / 3), np.quantile(m170, 2 / 3)
    helix = (m100 >= t100) & (m100 / t100 >= m170 / t170)
    strand = (m170 >= t170) & ~helix
    moment = np.where(helix, "helix-like", np.where(strand, "strand-like", "weak periodicity"))
    return {"moment": moment, "hydropathy": np.where(hyd >= 1.6, "TM-like (KD >= 1.6)", "soluble-like"),
            "mu100": m100, "mu170": m170}


def disorder_labels(seqs, sid, pos):
    """metapredict disorder score >= 0.5 at the target, or None if metapredict is absent."""
    try:
        import metapredict as meta
    except ImportError:
        return None, None
    scores = {}
    for i in np.unique(sid):
        scores[int(i)] = np.asarray(meta.predict_disorder(to_str(seqs[i] + N_SPECIAL)))
    d = np.array([scores[int(i)][p] for i, p in zip(sid, pos)])
    version = getattr(meta, "__version__", "unknown")
    return np.where(d >= 0.5, "disordered", "ordered"), version


def cluster_mean_ci(x: np.ndarray, cluster: np.ndarray, B: int = 1000, seed: int = 0) -> list[float]:
    """95% bootstrap interval for mean(x), resampling whole clusters (proteins)."""
    rng = np.random.default_rng(seed)
    u, inv = np.unique(cluster, return_inverse=True)
    s, c = np.bincount(inv, x, len(u)), np.bincount(inv, None, len(u))
    d = rng.integers(0, len(u), (B, len(u)))
    v = s[d].sum(1) / c[d].sum(1)
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]


def split_gains(labels: np.ndarray, sid: np.ndarray, gains: dict[str, np.ndarray]) -> dict:
    total = {k: float(g.sum()) for k, g in gains.items()}
    out = {}
    for lab in sorted(set(labels.tolist())):
        m = labels == lab
        row = {"targets": int(m.sum()), "fraction": float(m.mean())}
        for k, g in gains.items():
            row[k] = float(g[m].mean())
            row[f"{k}_ci95"] = cluster_mean_ci(g[m], sid[m])
            row[f"{k}_share_of_total"] = float(g[m].sum() / total[k])
        out[lab] = row
    return out


# ------------------------------------------------------------------ main
def run() -> dict:
    from .evaluate import protein_train_stream
    from .train import device

    log("fitting the 3-gram")
    flat, tg = protein_train_stream()
    table3 = ngram.KneserNey.fit(flat, tg, 3, 21, 20).table(3)
    bg = np.bincount(flat[tg], minlength=20) / len(tg)
    del flat, tg

    # validation: the original 20k sweep targets reproduce cache_protein's grid
    vseqs = [residues(s) for s in protein_subset("val", N_EVAL)]
    vparts = Parts(*long_context_targets(vseqs, SWEEP_N, SWEEP_SEED)[:2], table3, bg)
    lam, alpha = grid_cache(vparts)
    del vparts, vseqs
    # feature weights are estimated on training families, like the n-gram itself; mixing
    # weights and regularisation are chosen on validation families, like cache_protein
    trn = Parts(*packed_targets(load_protein("train"), TRAIN_N, FIT_SEED), table3, bg)
    val = Parts(*packed_targets(load_protein("val"), VAL_N, FIT_SEED), table3, bg)
    tests = {}
    for split in ("test_novel", "test_seen"):
        seqs = [residues(s) for s in protein_subset(split, N_EVAL)]
        ctx, tgt, tsid, tpos = long_context_targets(seqs, SWEEP_N, SWEEP_SEED)
        tests[split] = (Parts(ctx, tgt, table3, bg), seqs, tsid, tpos)
    log(f"targets ready: {len(trn.tgt)} train, {len(val.tgt)} validation; cache (lambda, alpha) = ({lam}, {alpha})")

    def cache_rows(p: Parts):
        return lam * p.p3 + (1 - lam) * p.cache(K, alpha)

    # local composition: EM over the 3-gram and caches of every window and smoothing, on validation
    names = [n for n, _ in val.components()]
    w_local = fit_mixture(np.stack([val.at(r) for _, r in val.components()], 1))
    # literal "mix in the lag counts": P(a | lag pattern) counted on training, EM weights on validation
    pat_trn, n_pat = lag_pattern(trn.ctx, PERIODIC_LAGS, HYDRO2)
    q = np.ones((n_pat, 20))
    np.add.at(q, (pat_trn, trn.tgt), 1)
    q /= q.sum(1, keepdims=True)
    pat_val, _ = lag_pattern(val.ctx, PERIODIC_LAGS, HYDRO2)
    w_lin = fit_mixture(np.stack([val.at(val.p3), val.at(val.cache(K, alpha)), q[pat_val, val.tgt]], 1))

    def feats(p: Parts, spec: str):
        f = []
        if "p2" in spec:
            f.append(lag_pattern(p.ctx, PERIODIC_LAGS, HYDRO2))
        if "p3" in spec:
            f.append(lag_pattern(p.ctx, PERIODIC_LAGS, HYDRO3))
        if "id" in spec:
            f += [(p.ctx[:, -L].astype(np.int64), 20) for L in ID_LAGS]
        if spec.startswith("lag"):  # single-lag scan: "lagC<L>" class, "lagI<L>" identity
            L = int(spec[4:])
            f.append((HYDRO2[p.ctx[:, -L]], 2) if spec[3] == "C" else (p.ctx[:, -L].astype(np.int64), 20))
        return f

    bases = {"cache": cache_rows, "local": lambda p: p.mix(w_local)}

    def loglinear(base: str, spec: str, l2s=L2_GRID, fit_on: Parts | None = None):
        src = fit_on or trn
        lb, f = np.log(bases[base](src)), feats(src, spec)
        lb_val, f_val = np.log(bases[base](val)), feats(val, spec)
        best = None
        for l2 in l2s:
            W = fit_loglinear(lb, f, src.tgt, l2)
            b = float(-target_bits(apply_loglinear(lb_val, f_val, W), val.tgt).mean())
            if best is None or b < best[0]:
                best = (b, l2, W)
        _, l2, W = best
        out = {"l2": l2, "val_bits": best[0], "val_bits_base": float(-np.log2(val.at(bases[base](val))).mean()),
               "n_params": int(sum(n * 20 for _, n in f))}
        for split, (p, *_rest) in tests.items():
            lp = target_bits(apply_loglinear(np.log(bases[base](p)), feats(p, spec), W), p.tgt)
            out[split] = float(-lp.mean())
            out[f"_{split}_logp"] = lp
        return out

    res = {"k": K, "cache_grid_on_val": [lam, alpha], "train_targets": int(len(trn.tgt)),
           "val_targets": int(len(val.tgt)), "lags": list(PERIODIC_LAGS), "hydrophobic": HYDROPHOBIC,
           "local_mixture_weights": {n: float(w) for n, w in zip(names, w_local) if w > 1e-3},
           "linear_periodic_mixture_weights": w_lin.tolist(), "baselines": {}, "neural": {}}
    B = res["baselines"]
    logps = {}
    for split, (p, *_rest) in tests.items():
        for name, rows in (("3-gram", p.p3), ("cache", cache_rows(p)), ("local", p.mix(w_local))):
            lp = np.log2(p.at(rows))
            B.setdefault(name, {})[split] = float(-lp.mean())
            logps[f"{split}/{name}"] = lp
        pat, _ = lag_pattern(p.ctx, PERIODIC_LAGS, HYDRO2)
        lp = np.log2(w_lin[0] * p.at(p.p3) + w_lin[1] * p.at(p.cache(K, alpha)) + w_lin[2] * q[pat, p.tgt])
        B.setdefault("cache+periodic (linear mix)", {})[split] = float(-lp.mean())
    log(f"linear mixtures done: local {B['local']['test_novel']:.4f}, "
        f"linear periodic {B['cache+periodic (linear mix)']['test_novel']:.4f}")
    for label, base, spec, kw in (("cache+periodic", "cache", "p2", {}),
                                  ("cache+periodic (3 classes)", "cache", "p3", {}),
                                  ("cache+periodic (weights fit on validation)", "cache", "p2",
                                   {"fit_on": val, "l2s": (1e-6,)}),
                                  ("local+periodic", "local", "p2", {}),
                                  ("local+periodic+lags 2-16", "local", "p2id", {})):
        r = loglinear(base, spec, **kw)
        for split in tests:
            logps[f"{split}/{label}"] = r.pop(f"_{split}_logp")
        B[label] = r
        log(f"{label}: held-out {r['test_novel']:.4f}, val {r['val_bits']:.4f} vs {r['val_bits_base']:.4f} "
            f"(L2 {r['l2']:g})")
    neural = RES / "cache" / "residual_neural.npz"  # reused across runs: the slow part on a shared GPU
    old = dict(np.load(neural)) if neural.exists() else {}
    dev = device()
    for split, (p, *_rest) in tests.items():
        for name in ("lstm", "gpt"):
            key = f"{split}/{name}"
            if key not in old or len(old[key]) != len(p.tgt):
                old[key] = neural_target_bits(ROOT / "checkpoints" / f"protein_{name}.pt", p.ctx, p.tgt, dev)
                np.savez(neural, **old)
            lp = old[key].astype(np.float64)
            res["neural"].setdefault(name, {})[split] = float(-lp.mean())
            logps[key] = lp
    log(f"neural: LSTM held-out {res['neural']['lstm']['test_novel']:.4f}")
    for split in tests:
        c, l = B["cache"][split], res["neural"]["lstm"][split]
        for name, r in B.items():
            r.setdefault("residual_closed", {})[split] = (c - r[split]) / (c - l)

    # single-lag scan: how much does one lag add on top of the cache model?
    # (regularised as strongly as validation chose for the four-lag periodic model)
    scan_l2 = (B["cache+periodic"]["l2"],)
    res["lag_scan_l2"] = scan_l2[0]
    scan = {}
    for L in SCAN_LAGS:
        row = {}
        for kind, tag in (("hydrophobic class", "C"), ("identity", "I")):
            r = loglinear("cache", f"lag{tag}{L}", l2s=scan_l2)
            row[kind] = {s: B["cache"][s] - r[s] for s in tests}
            row[kind]["val"] = r["val_bits_base"] - r["val_bits"]
        scan[L] = row
        log(f"lag {L}: class {1000 * row['hydrophobic class']['test_novel']:.2f} mb, "
            f"identity {1000 * row['identity']['test_novel']:.2f} mb")
    res["lag_scan_gain_over_cache"] = scan

    # where does the LSTM gain over the cache model? (held-out families)
    _, seqs, tsid, tpos = tests["test_novel"]
    lp = {k.split("/")[1]: v for k, v in logps.items() if k.startswith("test_novel/")}
    res.update(per_target(seqs, tsid, tpos, lp))
    np.savez(RES / "cache" / "residual_logp.npz", **{k: v.astype(np.float32) for k, v in logps.items()},
             sid=tsid, pos=tpos)
    return res


def per_target(seqs, sid, pos, lp: dict[str, np.ndarray]) -> dict:
    """Split per-target gains over the cache model by structure, hydropathy, residue and disorder."""
    tgt = np.array([seqs[i][p] for i, p in zip(sid, pos)])
    gains = {"lstm_minus_cache": lp["lstm"] - lp["cache"],
             "periodic_minus_cache": lp["cache+periodic"] - lp["cache"],
             "best_explicit_minus_cache": lp["local+periodic+lags 2-16"] - lp["cache"]}
    lab = structure_labels(seqs, sid, pos)
    split = {"moment": split_gains(lab["moment"], sid, gains),
             "hydropathy": split_gains(lab["hydropathy"], sid, gains),
             "target residue": split_gains(np.array([AMINO[t] for t in tgt]), sid, gains)}
    log("structure labels done, running metapredict")
    out = {}
    dis, ver = disorder_labels(seqs, sid, pos)
    if dis is not None:
        split["disorder (metapredict)"] = split_gains(dis, sid, gains)
        out["metapredict_version"] = ver
    out["per_target_test_novel"] = split
    out["per_target_mean_gain"] = {k: float(g.mean()) for k, g in gains.items()}
    return out


def splits_only() -> dict:
    """Recompute the per-target splits from results/cache/residual_logp.npz (no refitting)."""
    res = json.loads((RES / "residual_protein.json").read_text())
    z = np.load(RES / "cache" / "residual_logp.npz")
    seqs = [residues(s) for s in protein_subset("test_novel", N_EVAL)]
    lp = {k.split("/")[1]: z[k].astype(np.float64) for k in z.files if k.startswith("test_novel/")}
    res.update(per_target(seqs, z["sid"], z["pos"], lp))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits-only", action="store_true", help="redo only the per-target splits")
    args = ap.parse_args()
    (RES / "cache").mkdir(parents=True, exist_ok=True)
    res = splits_only() if args.splits_only else run()
    (RES / "residual_protein.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "per_target_test_novel"}, indent=2)[:6000])


if __name__ == "__main__":
    main()
