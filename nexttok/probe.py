"""Linear probes: is the reading frame linearly readable from the hidden state?

For each held-out target base, the hidden state that predicts it (the state
after reading the base before it, in the windows nexttok.evaluate scores) is
fed to a multinomial logistic regression over the 7 classes: non-coding and
+/- strand codon positions 1-3. Bases in two overlapping genes are left out.

Probes are trained on positions from the validation blocks and scored on
positions from the test blocks; neither was used to fit the networks' weights.
References on the same positions:
  majority        always the most common class
  8-gram context  one-hot of the 7 preceding bases, jointly (16,384 features)
                  and position by position (28 features): all an 8-gram sees
  untrained net   the same architecture at its random initialisation
  frame HMM       argmax of P(class | preceding bases) from nexttok.frame

  python -m nexttok.probe --genome ecoli     # writes results/frame/probe_ecoli.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .evaluate import DNA_CTX, dna_targets
from .frame import CACHE, CLASS_NAMES, N_HYP, OUT, load_checkpoint, scoring_windows
from .genome import GENOMES, checkpoint, use_genome
from .models import Config, build
from .train import device
from .vocab import N_SPECIAL, vocab_size

N_POS = 50_000          # probe positions drawn from each of the validation and test blocks
LAMBDAS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
CONTEXT = 7             # the 8-gram's context


# ----------------------------------------------------------------- positions
def probe_positions(split_id: int, seed: int):
    """A fixed sample of scored targets (>= 256 bases of context), overlap bases excluded."""
    sym, tg, labels, segs = dna_targets(split_id)
    keep = labels < N_HYP
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(tg[keep], size=min(N_POS, int(keep.sum())), replace=False))
    return sym, segs, pick


# ----------------------------------------------------------------- features
def layer_names(model) -> list[str]:
    if hasattr(model, "rnn"):
        return [f"layer{i + 1}" for i in range(model.rnn.num_layers)]
    return ["embed"] + [f"block{i + 1}" for i in range(len(model.blocks))]


def split_lstm(rnn: nn.LSTM) -> list[nn.LSTM]:
    """One single-layer LSTM per layer of a stacked one, sharing its weights."""
    layers = []
    for i in range(rnn.num_layers):
        layer = nn.LSTM(rnn.input_size if i == 0 else rnn.hidden_size, rnn.hidden_size, batch_first=True)
        layer.load_state_dict({f"{n}_l0": getattr(rnn, f"{n}_l{i}")
                               for n in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")})
        layers.append(layer.to(next(rnn.parameters()).device).eval())
    return layers


@torch.no_grad()
def hidden_states(model, x: torch.Tensor) -> list[torch.Tensor]:
    """Per-layer hidden states (B, T, D): LSTM layer outputs or the GPT residual stream."""
    if hasattr(model, "rnn"):
        if not hasattr(model, "_layers"):
            model._layers = split_lstm(model.rnn)
        h, outs = model.tok(x), []
        for layer in model._layers:
            h, _ = layer(h)
            outs.append(h)
        return outs
    T = x.shape[1]
    h = model.tok(x) + model.pos(torch.arange(T, device=x.device))
    outs = [h]
    for b in model.blocks:
        h = b(h)
        outs.append(h)
    return outs


@torch.no_grad()
def extract(model, sym: np.ndarray, segs, pick: np.ndarray, dev) -> dict[str, np.ndarray]:
    """Hidden state predicting each picked target, per layer, in the order of `pick`."""
    names = layer_names(model)
    feats = {n: [] for n in names}
    pos_all = []
    for W, p in scoring_windows(sym, segs):
        keep = np.isin(p, pick).reshape(W.shape[0], -1)
        for i in range(0, len(W), 32):
            k = keep[i: i + 32]
            if not k.any():
                continue
            x = torch.from_numpy(W[i: i + 32] + N_SPECIAL).to(dev)
            hs = hidden_states(model, x)
            for n, h in zip(names, hs):
                h = h[:, DNA_CTX - 1: W.shape[1] - 1].float().cpu().numpy()  # column t-1 predicts t
                feats[n].append(h[k])
        pos_all.append(p[keep.reshape(-1)])
    order = np.argsort(np.concatenate(pos_all))
    assert np.array_equal(np.concatenate(pos_all)[order], pick)
    return {n: np.concatenate(v)[order] for n, v in feats.items()}


def context_codes(sym: np.ndarray, pick: np.ndarray, k: int = CONTEXT) -> np.ndarray:
    """(n, k) preceding bases, oldest first."""
    return sym[pick[:, None] + np.arange(-k, 0)]


# ----------------------------------------------------------------- probes
class Linear(nn.Module):
    def __init__(self, n_in: int, onehot: bool):
        super().__init__()
        self.onehot = onehot
        self.w = nn.Embedding(n_in, N_HYP) if onehot else nn.Linear(n_in, N_HYP, bias=False)
        nn.init.zeros_(self.w.weight)
        self.b = nn.Parameter(torch.zeros(N_HYP))

    def forward(self, x):
        return self.w(x) + self.b


def fit_probe(x: torch.Tensor, y: torch.Tensor, lam: float, onehot: bool, n_in: int) -> Linear:
    """Full-batch L-BFGS on cross-entropy + lam * ||W||^2."""
    m = Linear(n_in, onehot)
    opt = torch.optim.LBFGS(m.parameters(), max_iter=500, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(m(x), y) + lam * m.w.weight.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return m


@torch.no_grad()
def scores(m: Linear, x: torch.Tensor, y: torch.Tensor, prior_bits: float) -> dict:
    logits = m(x)
    ce = float(F.cross_entropy(logits, y)) / math.log(2)
    guess = logits.argmax(1)
    coding = y > 0
    return {"accuracy": float((guess == y).float().mean()),
            "frame_accuracy_on_coding": float((guess[coding] == y[coding]).float().mean()),
            "bits": ce, "class_info_bits": prior_bits - ce,
            "recall": {CLASS_NAMES[c]: float((guess[y == c] == c).float().mean()) for c in range(N_HYP)}}


def probe(x_tr, y_tr, x_te, y_te, prior_bits: float, onehot: bool = False, n_in: int | None = None) -> dict:
    """Pick the L2 strength on the last fifth of the training positions, refit on all, score on test."""
    if not onehot:
        mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-6
        x_tr, x_te = (x_tr - mu) / sd, (x_te - mu) / sd
    n_in = n_in or x_tr.shape[1]
    cut = int(0.8 * len(y_tr))  # positions are sorted, so this holds out whole stretches of genome
    held = {lam: float(F.cross_entropy(fit_probe(x_tr[:cut], y_tr[:cut], lam, onehot, n_in)(x_tr[cut:]), y_tr[cut:]))
            for lam in LAMBDAS}
    lam = min(held, key=held.get)
    m = fit_probe(x_tr, y_tr, lam, onehot, n_in)
    return {"lambda": lam, **scores(m, x_te, y_te, prior_bits)}


# ----------------------------------------------------------------- experiment
def run(genome: str) -> dict:
    torch.manual_seed(0)
    with use_genome(genome):
        sym_v, segs_v, pick_v = probe_positions(1, seed=0)
        sym_t, segs_t, pick_t = probe_positions(2, seed=1)
        from .evaluate import load_dna
        lab = load_dna()["labels"].astype(np.int64)
    y_tr, y_te = torch.from_numpy(lab[pick_v]), torch.from_numpy(lab[pick_t])
    prior = np.bincount(lab[pick_v], minlength=N_HYP) / len(pick_v)
    prior_bits = float(-np.log2(prior[lab[pick_t]]).mean())
    out = {"genome": genome, "n_train": int(len(pick_v)), "n_test": int(len(pick_t)),
           "prior_bits": prior_bits, "majority_accuracy": float((lab[pick_t] == prior.argmax()).mean()),
           "probes": {}}

    # what an 8-gram can see
    c_tr, c_te = context_codes(sym_v, pick_v), context_codes(sym_t, pick_t)
    joint = lambda c: torch.from_numpy((c * 4 ** np.arange(CONTEXT - 1, -1, -1)).sum(1))
    out["probes"]["8-gram context, joint one-hot"] = probe(joint(c_tr), y_tr, joint(c_te), y_te, prior_bits,
                                                           onehot=True, n_in=4 ** CONTEXT)
    split_hot = lambda c: torch.from_numpy(np.eye(4, dtype=np.float32)[c].reshape(len(c), -1))
    out["probes"]["8-gram context, per-position one-hot"] = probe(split_hot(c_tr), y_tr, split_hot(c_te), y_te,
                                                                  prior_bits)

    # explicit frame inference by the GeneMark-style HMM on the same positions
    cache_path = CACHE / f"frame_{genome}.npz"
    if cache_path.exists():
        c = np.load(cache_path)
        with use_genome(genome):
            _, tg, _, _ = dna_targets(2)
        st = c["test/frame_hmm/state"][np.searchsorted(tg, pick_t)]
        guess = st[:, :N_HYP].argmax(1)
        yt = lab[pick_t]
        out["frame_hmm_argmax"] = {"accuracy": float((guess == yt).mean()),
                                   "frame_accuracy_on_coding": float((guess[yt > 0] == yt[yt > 0]).mean())}

    dev = device()
    best_lstm = None
    for name in ("lstm", "gpt"):
        trained = load_checkpoint(checkpoint(genome, name), dev)
        untrained = build(name, Config(vocab=vocab_size("dna"), max_len=512)).to(dev).eval()
        for tag, model in (("", trained), (" (untrained)", untrained)):
            f_tr = extract(model, sym_v, segs_v, pick_v, dev)
            f_te = extract(model, sym_t, segs_t, pick_t, dev)
            for layer in f_tr:
                key = f"{name.upper() if name == 'gpt' else 'LSTM'} {layer}{tag}"
                r = probe(torch.from_numpy(f_tr[layer]), y_tr, torch.from_numpy(f_te[layer]), y_te, prior_bits)
                out["probes"][key] = r
                print(key, f"acc {r['accuracy']:.3f}  info {r['class_info_bits']:.3f} bits", flush=True)
                if name == "lstm" and not tag and (best_lstm is None or r["accuracy"] > best_lstm[1]["accuracy"]):
                    best_lstm = (layer, r, f_tr[layer], f_te[layer])
            del f_tr, f_te

    # does the LSTM gain over the 8-gram follow from knowing the frame?
    if cache_path.exists() and best_lstm is not None:
        layer, r, ftr, fte = best_lstm
        mu, sd = ftr.mean(0), ftr.std(0) + 1e-6
        m = fit_probe(torch.from_numpy((ftr - mu) / sd), y_tr, r["lambda"], False, ftr.shape[1])
        with torch.no_grad():
            guess = m(torch.from_numpy((fte - mu) / sd)).argmax(1).numpy()
        c = np.load(cache_path)
        with use_genome(genome):
            _, tg, _, _ = dna_targets(2)
        at = np.searchsorted(tg, pick_t)
        best_ng = max((k for k in c.files if k.endswith("-gram")), key=lambda k: c[k].mean())
        gain = c["test/lstm"][at] - c[best_ng][at]  # log2 P(LSTM) - log2 P(n-gram), bits saved
        yt = lab[pick_t]
        out["lstm_gain_by_probe"] = {"layer": layer, "ngram": best_ng.split("/")[1]}
        for cname, cls in (("+ codon 3", 3), ("− codon 1", 4), ("non-coding", 0)):
            ok, bad = (yt == cls) & (guess == yt), (yt == cls) & (guess != yt)
            out["lstm_gain_by_probe"][cname] = {
                "frame_decoded": {"n": int(ok.sum()), "bits_saved": float(gain[ok].mean())},
                "frame_missed": {"n": int(bad.sum()), "bits_saved": float(gain[bad].mean())}}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome", choices=sorted(GENOMES), default="ecoli")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    res = run(args.genome)
    (OUT / f"probe_{args.genome}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "probes"}, indent=1))


if __name__ == "__main__":
    main()
