"""Evaluation suite. Every model is scored on the *same* target positions.

Parts (run independently, each writes results/<part>_<kind>.json):
  ngram     Kneser-Ney ladder (protein orders 1-6, DNA orders 1-12)
  neural    GPT and LSTM checkpoints, full available context
  context   truncated-context sweep: predict x_i from only the last k symbols
  audit     shared 10-mer rate between each test split and the training data
  generate  sample proteins from the GPT and compare them with natural ones

Per-target log-probabilities go to results/cache/ (git-ignored) so that the
report can slice them by position, residue, or codon phase.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from . import ngram
from .data import load_dna, load_protein, protein_subset, segments
from .models import Config, build
from .train import device
from .vocab import AMINO, BOS, EOS, N_SPECIAL, symbol_ids, vocab_size

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
CACHE = RES / "cache"
N_EVAL = 3000          # proteins per test split
DNA_CTX = 256          # every scored DNA position has >= this much context
SWEEP_K = [1, 2, 4, 8, 16, 32, 64, 128, 256]
SWEEP_N = 20000        # targets per split in the context sweep
PROTEIN_MAX_ORDER = 6
DNA_ORDERS = [1, 2, 3, 4, 6, 8, 10, 12]


def summarise(logp: np.ndarray, correct: np.ndarray) -> dict:
    bits = float(-logp.mean())
    return {"bits": bits, "perplexity": 2 ** bits, "accuracy": float(correct.mean()), "n": int(len(logp))}


# ----------------------------------------------------------------- targets
def protein_targets(split: str):
    """Flat padded symbol stream + target positions for the eval subset.

    Symbols: residues 0..19, BOS/pad = 20. Each sequence is preceded by
    PROTEIN_MAX_ORDER-1 pads, so every n-gram order sees a BOS-padded context.
    Returns (flat, targets, seq_id, pos_in_seq, seqs).
    """
    seqs = protein_subset(split, N_EVAL)
    pad = PROTEIN_MAX_ORDER - 1
    flat, tg, sid, pos = [], [], [], []
    off = 0
    for i, s in enumerate(seqs):
        flat.append(np.full(pad, 20, np.int64))
        flat.append(s.astype(np.int64) - N_SPECIAL)
        tg.append(off + pad + np.arange(len(s)))
        sid.append(np.full(len(s), i))
        pos.append(np.arange(len(s)))
        off += pad + len(s)
    return np.concatenate(flat), np.concatenate(tg), np.concatenate(sid), np.concatenate(pos), seqs


def protein_train_stream():
    d = load_protein("train")
    sym = d["tokens"].astype(np.int64) - N_SPECIAL  # BOS -> -2, EOS -> -1
    pad = PROTEIN_MAX_ORDER - 1
    # rebuild as [pad*5, seq] blocks without a python loop
    lengths = d["lengths"]
    block = lengths + pad
    starts = np.concatenate([[0], np.cumsum(block)[:-1]])
    flat = np.full(int(block.sum()), 20, np.int64)
    res_mask = (sym >= 0)
    dst = (np.repeat(starts + pad, lengths) + (np.arange(int(lengths.sum())) - np.repeat(np.cumsum(lengths) - lengths, lengths)))
    flat[dst] = sym[res_mask]
    return flat, dst


def dna_targets(split_id: int = 2):
    d = load_dna()
    sym = d["tokens"].astype(np.int64) - N_SPECIAL
    segs = segments(d["split"] == split_id)
    tg = np.concatenate([np.arange(a + DNA_CTX, b) for a, b in segs])
    return sym, tg, d["labels"][tg], segs


def dna_train_stream(max_ctx: int):
    d = load_dna()
    sym = d["tokens"].astype(np.int64) - N_SPECIAL
    parts, tg, off = [], [], 0
    for a, b in segments(d["split"] == 0):
        for s in (sym[a:b], 3 - sym[a:b][::-1]):  # forward + reverse complement
            parts.append(s)
            tg.append(off + np.arange(max_ctx, len(s)))
            off += len(s)
    return np.concatenate(parts), np.concatenate(tg)


# ----------------------------------------------------------------- n-grams
def run_ngram(kind: str) -> dict:
    out, cache = {}, {}
    if kind == "protein":
        flat, tg = protein_train_stream()
        model = ngram.KneserNey.fit(flat, tg, PROTEIN_MAX_ORDER, 21, 20)
        orders = range(1, PROTEIN_MAX_ORDER + 1)
        for split in ("test_seen", "test_novel"):
            ef, et, _, _, _ = protein_targets(split)
            out[split] = {"uniform": summarise(np.full(len(et), -math.log2(20)), np.zeros(len(et)))}
            for o in orders:
                logp, pred, _ = ngram.score(model.table(o), o, ef, et, 21, 20)
                out[split][f"{o}-gram"] = summarise(logp, pred == ef[et])
                cache[f"{split}/{o}-gram"] = logp.astype(np.float32)
    else:
        flat, tg = dna_train_stream(max(DNA_ORDERS) - 1)
        model = ngram.KneserNey.fit(flat, tg, max(DNA_ORDERS), 4, 4)
        ef, et, labels, _ = dna_targets()
        out["test"] = {"uniform": summarise(np.full(len(et), -2.0), np.zeros(len(et)))}
        for o in DNA_ORDERS:
            logp, pred, _ = ngram.score(model.table(o), o, ef, et, 4, 4)
            out["test"][f"{o}-gram"] = summarise(logp, pred == ef[et])
            cache[f"test/{o}-gram"] = logp.astype(np.float32)
    save_cache(kind, cache)
    return out


# ----------------------------------------------------------------- neural
def load_model(kind: str, name: str, dev):
    ck = torch.load(ROOT / "checkpoints" / f"{kind}_{name}.pt", map_location="cpu")
    m = build(ck["model"], Config(**ck["cfg"]))
    m.load_state_dict(ck["state"])
    return m.to(dev).eval()


@torch.no_grad()
def symbol_logprobs(model, x: torch.Tensor, kind: str) -> torch.Tensor:
    """log2 P over the symbol alphabet only, shape (B, T, |alphabet|)."""
    logits = model(x)[..., symbol_ids(kind)]
    return torch.log_softmax(logits.float(), -1) / math.log(2)


def run_neural(kind: str) -> dict:
    dev = device()
    out, cache = {}, {}
    for name in ("gpt", "lstm"):
        model = load_model(kind, name, dev)
        if kind == "protein":
            for split in ("test_seen", "test_novel"):
                _, _, _, _, seqs = protein_targets(split)
                order = np.argsort([len(s) for s in seqs])
                lp = [None] * len(seqs)
                pr = [None] * len(seqs)
                for i in range(0, len(order), 32):
                    idx = order[i: i + 32]
                    L = max(len(seqs[j]) for j in idx)
                    x = torch.zeros(len(idx), L, dtype=torch.long)
                    for r, j in enumerate(idx):
                        s = torch.from_numpy(seqs[j].astype(np.int64))
                        x[r, 0] = BOS
                        x[r, 1: len(s)] = s[:-1]
                    p = symbol_logprobs(model, x.to(dev), kind).cpu()
                    for r, j in enumerate(idx):
                        t = torch.from_numpy(seqs[j].astype(np.int64)) - N_SPECIAL
                        pj = p[r, : len(t)]
                        lp[j] = pj.gather(1, t[:, None])[:, 0].numpy()
                        pr[j] = (pj.argmax(1) == t).numpy()
                logp, corr = np.concatenate(lp), np.concatenate(pr)
                out.setdefault(split, {})[name] = summarise(logp, corr)
                cache[f"{split}/{name}"] = logp.astype(np.float32)
        else:
            sym, tg, _, segs = dna_targets()
            logp, corr = dna_neural_scores(model, sym, segs, dev)
            out.setdefault("test", {})[name] = summarise(logp, corr)
            cache[f"test/{name}"] = logp.astype(np.float32)
    save_cache(kind, cache)
    return out


@torch.no_grad()
def dna_neural_scores(model, sym, segs, dev, T: int = 512):
    """Windows of length T, stride T-DNA_CTX; score each window's tail."""
    step = T - DNA_CTX
    lps, cs = [], []
    for a, b in segs:
        starts = list(range(a, b - DNA_CTX, step))
        for i in range(0, len(starts), 32):
            batch = [sym[s: min(s + T, b)] for s in starts[i: i + 32]]
            L = max(len(w) for w in batch)
            x = torch.zeros(len(batch), L, dtype=torch.long)
            for r, w in enumerate(batch):
                x[r, : len(w)] = torch.from_numpy(w + N_SPECIAL)
            # position j predicts token j+1, so prepend nothing and read j-1
            p = symbol_logprobs(model, x.to(dev), "dna").cpu()
            for r, w in enumerate(batch):
                tgt = torch.from_numpy(w[DNA_CTX:])
                pj = p[r, DNA_CTX - 1: len(w) - 1]
                lps.append(pj.gather(1, tgt[:, None])[:, 0].numpy())
                cs.append((pj.argmax(1) == tgt).numpy())
    return np.concatenate(lps), np.concatenate(cs)


# ----------------------------------------------------------------- context sweep
def sweep_targets(kind: str, split: str):
    """Targets with >= max(SWEEP_K) symbols of preceding context."""
    rng = np.random.default_rng(7)
    K = max(SWEEP_K)
    if kind == "protein":
        seqs = protein_subset(split, N_EVAL)
        cand = [(i, p) for i, s in enumerate(seqs) for p in range(K, len(s))]
        pick = rng.choice(len(cand), size=min(SWEEP_N, len(cand)), replace=False)
        ctx = np.stack([seqs[cand[c][0]][cand[c][1] - K: cand[c][1]] for c in pick]).astype(np.int64)
        tgt = np.array([seqs[cand[c][0]][cand[c][1]] for c in pick], dtype=np.int64)
        return ctx, tgt
    sym, tg, _, _ = dna_targets(1 if split == "val" else 2)
    pick = rng.choice(tg, size=SWEEP_N, replace=False)
    ctx = np.stack([sym[p - K: p] for p in pick]) + N_SPECIAL
    return ctx.astype(np.int64), (sym[pick] + N_SPECIAL).astype(np.int64)


def run_context(kind: str) -> dict:
    dev = device()
    splits = ("test_seen", "test_novel") if kind == "protein" else ("test",)
    C, V = (21, 20) if kind == "protein" else (4, 4)
    N = PROTEIN_MAX_ORDER if kind == "protein" else max(DNA_ORDERS)
    flat, tg = protein_train_stream() if kind == "protein" else dna_train_stream(N - 1)
    ng = ngram.KneserNey.fit(flat, tg, N, C, V)
    del flat, tg
    out = {}
    for split in splits:
        ctx, tgt = sweep_targets(kind, split)
        t = torch.from_numpy(tgt - N_SPECIAL)
        out[split] = {}
        for name in ("gpt", "lstm"):
            model = load_model(kind, name, dev)
            res = {}
            for k in SWEEP_K:
                lps = []
                for i in range(0, len(ctx), 512):
                    x = torch.from_numpy(ctx[i: i + 512, -k:]).to(dev)
                    lps.append(symbol_logprobs(model, x, kind)[:, -1].cpu())
                lp = torch.cat(lps).gather(1, t[:, None])[:, 0]
                res[k] = float(-lp.mean())
            out[split][name] = res
        # n-gram reference on the same targets
        stream = np.concatenate([ctx - N_SPECIAL, (tgt - N_SPECIAL)[:, None]], 1)
        flat_e = stream.reshape(-1)
        pos = np.arange(len(stream)) * stream.shape[1] + stream.shape[1] - 1
        out[split]["ngram"] = {}
        for o in (range(1, N + 1) if kind == "protein" else DNA_ORDERS):
            logp, _, _ = ngram.score(ng.table(o), o, flat_e, pos, C, V)
            out[split]["ngram"][o - 1] = float(-logp.mean())  # keyed by context length
    return out


# ----------------------------------------------------------------- composition control
def run_cache(kind: str) -> dict:
    """Is the neural models' long-context gain just per-sequence composition?

    Baseline: a short n-gram linearly mixed with a smoothed unigram of the last
    k symbols ("cache" model). Mixing weight and smoothing are tuned per k on
    the *validation* split, then scored on the same targets as the sweep.
    """
    C, V = (21, 20) if kind == "protein" else (4, 4)
    N = PROTEIN_MAX_ORDER if kind == "protein" else max(DNA_ORDERS)
    base_order = 3 if kind == "protein" else 8  # best fixed order on test (see ngram_*.json)
    flat, tg = protein_train_stream() if kind == "protein" else dna_train_stream(N - 1)
    ng = ngram.KneserNey.fit(flat, tg, N, C, V)
    bg = np.bincount(flat[tg], minlength=V) / len(tg)
    del flat, tg
    tables = {o: ng.table(o) for o in range(1, base_order + 1)}

    def parts(split):
        ctx, tgt = sweep_targets(kind, split)
        sym, t = ctx - N_SPECIAL, tgt - N_SPECIAL
        stream = np.concatenate([sym, t[:, None]], 1)
        pos = np.arange(len(stream)) * stream.shape[1] + stream.shape[1] - 1
        out = {}
        for k in SWEEP_K:
            o = min(base_order, k + 1)
            logp, _, _ = ngram.score(tables[o], o, stream.reshape(-1), pos, C, V)
            hits = (sym[:, -k:] == t[:, None]).sum(1)
            out[k] = (2.0 ** logp, hits, bg[t])
        return out

    lams = np.linspace(0, 1, 21)
    alphas = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]

    def bits(p_ng, hits, b, k, lam, a):
        return float(-np.log2(lam * p_ng + (1 - lam) * (hits + a * b) / (k + a)).mean())

    val = parts("val")
    best = {}
    for k in SWEEP_K:
        grid = [(bits(*val[k], k, l, a), l, a) for l in lams for a in alphas]
        _, l, a = min(grid)
        best[k] = (float(l), float(a))
    splits = ("test_seen", "test_novel") if kind == "protein" else ("test",)
    res = {"base_order": base_order, "tuned_on_val": {str(k): v for k, v in best.items()}}
    for split in splits:
        te = parts(split)
        res[split] = {"ngram_plus_cache": {k: bits(*te[k], k, *best[k]) for k in SWEEP_K},
                      "ngram_alone": {k: float(-np.log2(te[k][0]).mean()) for k in SWEEP_K}}
    return res


# ----------------------------------------------------------------- audit
def kmer_codes(seqs, k: int = 10) -> np.ndarray:
    codes = []
    for s in seqs:
        s = s.astype(np.int64) - N_SPECIAL
        if len(s) < k:
            continue
        w = np.lib.stride_tricks.sliding_window_view(s, k)
        codes.append((w * (20 ** np.arange(k - 1, -1, -1))).sum(1))
    return np.concatenate(codes)


HASH_BITS = 32  # 512 MB bitset; false-positive rate is measured, not assumed


def _hash(codes: np.ndarray) -> np.ndarray:
    return (codes.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(64 - HASH_BITS)


def train_kmers(k: int = 10) -> np.ndarray:
    """Bitset of hashed training k-mers, built in chunks to bound memory."""
    d = load_protein("train")
    bits = np.zeros(2 ** HASH_BITS // 8, dtype=np.uint8)
    offs, lens = d["offsets"], d["lengths"]
    for i in range(0, len(offs), 20000):
        seqs = [d["tokens"][o + 1: o + 1 + n] for o, n in zip(offs[i: i + 20000], lens[i: i + 20000])]
        h = _hash(kmer_codes(seqs, k))
        np.bitwise_or.at(bits, (h >> np.uint64(3)).astype(np.int64), (np.uint8(1) << (h & np.uint64(7)).astype(np.uint8)))
    return bits


def shared_rate(codes: np.ndarray, bits: np.ndarray) -> float:
    h = _hash(codes)
    hit = (bits[(h >> np.uint64(3)).astype(np.int64)] >> (h & np.uint64(7)).astype(np.uint8)) & 1
    return float(hit.mean())


def run_audit(kind: str) -> dict:
    assert kind == "protein"
    ref = train_kmers()
    rng = np.random.default_rng(0)
    out = {}
    for split in ("test_seen", "test_novel"):
        seqs = protein_subset(split, N_EVAL)
        per_seq = [shared_rate(kmer_codes([s]), ref) for s in seqs]
        out[split] = {"kmer10_shared_with_train": shared_rate(kmer_codes(seqs), ref),
                      "frac_seqs_over_50pct_shared": float(np.mean(np.array(per_seq) > 0.5))}
    # random 10-mers essentially never truly occur in 73M residues, so their hit
    # rate is the bitset's false-positive rate
    rand = [(rng.integers(0, 20, 300) + N_SPECIAL).astype(np.uint8) for _ in range(3000)]
    out["random_iid_false_positive_rate"] = shared_rate(kmer_codes(rand), ref)
    return out


# ----------------------------------------------------------------- generation
@torch.no_grad()
def sample_proteins(model, n: int, dev, temp: float = 1.0, max_len: int = 512, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.full((n, 1), BOS, dtype=torch.long)
    done = torch.zeros(n, dtype=torch.bool)
    for _ in range(max_len):
        logits = model(x.to(dev))[:, -1].float().cpu() / temp
        logits[:, [0, BOS]] = -float("inf")
        nxt = torch.multinomial(torch.softmax(logits, -1), 1, generator=g)[:, 0]
        nxt[done] = EOS
        done |= nxt == EOS
        x = torch.cat([x, nxt[:, None]], 1)
        if done.all():
            break
    seqs = []
    for row in x[:, 1:].numpy():
        stop = np.flatnonzero(row == EOS)
        seqs.append(row[: stop[0] if len(stop) else len(row)].astype(np.uint8))
    return seqs


def profile(seqs) -> dict:
    seqs = [s for s in seqs if len(s) >= 10]
    allres = np.concatenate(seqs).astype(np.int64) - N_SPECIAL
    comp = np.bincount(allres, minlength=20) / len(allres)
    longest = []
    for s in seqs:
        change = np.flatnonzero(np.diff(s.astype(np.int64)) != 0)
        b = np.concatenate([[-1], change, [len(s) - 1]])
        longest.append(int(np.diff(b).max()))
    return {"n": len(seqs), "length_mean": float(np.mean([len(s) for s in seqs])),
            "length_median": float(np.median([len(s) for s in seqs])),
            "composition": comp.tolist(),
            "frac_homopolymer_run_ge6": float(np.mean(np.array(longest) >= 6))}


def js(p, q) -> float:
    p, q = np.asarray(p) + 1e-12, np.asarray(q) + 1e-12
    m = (p + q) / 2
    return float(0.5 * (p * np.log2(p / m)).sum() + 0.5 * (q * np.log2(q / m)).sum())


def run_generate(kind: str) -> dict:
    assert kind == "protein"
    dev = device()
    model = load_model("protein", "gpt", dev)
    ref = train_kmers()
    natural = protein_subset("test_novel", 500, seed=3)
    gen = []
    for b in range(2):  # 200 samples; no KV cache, so sampling is O(L^2)
        gen += sample_proteins(model, 100, dev, seed=b)
    rng = np.random.default_rng(0)
    d = load_protein("train")
    bg = np.bincount(d["tokens"][d["tokens"] >= N_SPECIAL].astype(np.int64) - N_SPECIAL, minlength=20)
    bg = bg / bg.sum()
    iid = [(rng.choice(20, size=len(s), p=bg) + N_SPECIAL).astype(np.uint8) for s in natural]
    out = {}
    for name, seqs in (("gpt_samples", gen), ("natural_test_novel", natural), ("iid_composition_matched", iid)):
        p = profile(seqs)
        p["composition_js_vs_train"] = js(p["composition"], bg)
        p["kmer10_shared_with_train"] = shared_rate(kmer_codes([s for s in seqs if len(s) >= 10]), ref)
        out[name] = p
    (RES / "generated_proteins.fasta").write_text(
        "".join(f">gpt_sample_{i}\n{''.join(AMINO[c - N_SPECIAL] for c in s)}\n" for i, s in enumerate(gen)))
    return out


# ----------------------------------------------------------------- io
def save_cache(kind: str, arrays: dict) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{kind}_logp.npz"
    old = dict(np.load(path)) if path.exists() else {}
    old.update(arrays)
    np.savez(path, **old)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["ngram", "neural", "context", "cache", "audit", "generate"])
    ap.add_argument("--kind", choices=["protein", "dna"], required=True)
    args = ap.parse_args()
    fn = {"ngram": run_ngram, "neural": run_neural, "context": run_context,
          "cache": run_cache, "audit": run_audit, "generate": run_generate}[args.part]
    res = fn(args.kind)
    RES.mkdir(exist_ok=True)
    (RES / f"{args.part}_{args.kind}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2)[:3000])


if __name__ == "__main__":
    main()
