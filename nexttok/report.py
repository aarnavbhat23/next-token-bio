"""Turn results/<group>/*.json and results/cache/<group>.npz into figures,
results/intervals.json and results/summary.md.

Protein numbers pool the family splits (an n-gram has one value per split) and
split x training seed (neural models); DNA numbers pool training seeds on the
one fixed block split. An interval "m ± s" is the mean and sample sd across
those runs. The paired contrasts the README leans on (GPT vs LSTM, the LSTM's
gain beyond composition, codon-position gains) also get, for every run, a 95%
bootstrap interval that resamples held-out families or 5 kb genome chunks: the
units within which test targets are correlated.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import data
from .evaluate import CACHE, DNA_ORDERS, PROTEIN_MAX_ORDER, SWEEP_K, dna_targets, protein_targets
from .vocab import AMINO

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = RES / "figures"
K = str(max(SWEEP_K))
SPLIT_NAMES = {"test_seen": "seen families", "test_novel": "held-out families", "test": "held-out blocks"}
CODON = ["non-coding", "+ codon 1", "+ codon 2", "+ codon 3", "− codon 1", "− codon 2", "− codon 3"]

# reference palette, first three categorical slots (validated all-pairs)
GPT, LSTM, NGRAM = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 2,
})


# --------------------------------------------------------------- loading
def load(group: str, part: str):
    p = RES / group / f"{part}.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_groups() -> tuple[dict, dict]:
    """({split_seed: {part: json}}, {part: json}) for protein and DNA."""
    parts = ("ngram", "neural", "context", "cache", "audit", "generate")
    splits = sorted(int(p.name[len("protein_split"):]) for p in RES.glob("protein_split*") if (p / "ngram.json").exists())
    prot = {s: {p: load(f"protein_split{s}", p) for p in parts} for s in splits}
    dna = {p: load("dna", p) for p in parts}
    return prot, dna


def seeds_of(d: dict, name: str) -> list[int]:
    return sorted(int(m.group(1)) for k in d if (m := re.fullmatch(rf"{name}/seed(\d+)", k)))


def npz(group: str) -> dict:
    return dict(np.load(CACHE / f"{group}.npz"))


# --------------------------------------------------------------- statistics
def stats(vals) -> dict:
    v = np.asarray(vals, dtype=np.float64)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": int(len(v)),
            "min": float(v.min()), "max": float(v.max()), "values": [float(x) for x in v]}


def pm(st: dict, d: int = 3) -> str:
    return f"{st['mean']:.{d}f} ± {st['sd']:.{d}f}" if st["n"] > 1 else f"{st['mean']:.{d}f}"


def cluster_ci(diff: np.ndarray, cluster: np.ndarray, B: int = 2000, seed: int = 0) -> dict:
    """Mean of `diff` with a 95% bootstrap interval that resamples whole clusters."""
    _, inv = np.unique(cluster, return_inverse=True)
    s, n = np.bincount(inv, diff.astype(np.float64)), np.bincount(inv).astype(np.float64)
    w = np.random.default_rng(seed).multinomial(len(s), np.full(len(s), 1 / len(s)), size=B)
    boot = (w @ s) / (w @ n)
    return {"est": float(diff.mean()), "lo": float(np.percentile(boot, 2.5)),
            "hi": float(np.percentile(boot, 97.5)), "clusters": int(len(s))}


# --------------------------------------------------------------- aggregation
def protein_intervals(prot: dict) -> dict:
    out = {"splits": sorted(prot), "ladder": {}, "accuracy": {}, "best_ngram_order": {}, "composition": {},
           "contrasts": {}, "runs": []}
    for split in ("test_seen", "test_novel"):
        ng = {s: prot[s]["ngram"][split] for s in prot}
        nn = {s: prot[s]["neural"][split] for s in prot}
        names = ["uniform"] + [f"{o}-gram" for o in range(1, PROTEIN_MAX_ORDER + 1)]
        lad = {n: stats([ng[s][n]["bits"] for s in prot]) for n in names}
        acc = {n: stats([ng[s][n]["accuracy"] for s in prot]) for n in names}
        best = {s: min((n for n in names if n != "uniform"), key=lambda n: ng[s][n]["bits"]) for s in prot}
        lad["best n-gram"] = stats([ng[s][best[s]]["bits"] for s in prot])
        acc["best n-gram"] = stats([ng[s][best[s]]["accuracy"] for s in prot])
        out["best_ngram_order"][split] = {f"split{s}": best[s] for s in prot}
        for m in ("gpt", "lstm"):
            lad[m] = stats([nn[s][f"{m}/seed{t}"]["bits"] for s in prot for t in seeds_of(nn[s], m)])
            acc[m] = stats([nn[s][f"{m}/seed{t}"]["accuracy"] for s in prot for t in seeds_of(nn[s], m)])
        out["ladder"][split], out["accuracy"][split] = lad, acc

        # composition control at the longest context, per split and per run
        comp = {"ngram_alone": stats([prot[s]["cache"][split]["ngram_alone"][K] for s in prot]),
                "ngram_plus_cache": stats([prot[s]["cache"][split]["ngram_plus_cache"][K] for s in prot])}
        for m in ("gpt", "lstm"):
            rows = [(s, t, prot[s]["context"][split][f"{m}/seed{t}"][K]) for s in prot
                    for t in seeds_of(prot[s]["context"][split], m)]
            alone = {s: prot[s]["cache"][split]["ngram_alone"][K] for s in prot}
            cached = {s: prot[s]["cache"][split]["ngram_plus_cache"][K] for s in prot}
            comp[m] = stats([v for _, _, v in rows])
            comp[f"{m}_gain_over_ngram"] = stats([alone[s] - v for s, _, v in rows])
            comp[f"{m}_residual_beyond_composition"] = stats([cached[s] - v for s, _, v in rows])
            comp[f"{m}_fraction_explained_by_composition"] = stats(
                [(alone[s] - cached[s]) / (alone[s] - v) for s, _, v in rows])
        comp["composition_gain"] = stats([prot[s]["cache"][split]["ngram_alone"][K]
                                          - prot[s]["cache"][split]["ngram_plus_cache"][K] for s in prot])
        out["composition"][split] = comp

    # paired contrasts with per-run family bootstraps (held-out families)
    gap, resid, gain1 = [], [], []
    for s in prot:
        c = npz(f"protein_split{s}")
        one = prot[s]["ngram"]["test_novel"]["1-gram"]["bits"]
        for t in seeds_of(prot[s]["neural"]["test_novel"], "lstm"):
            r = {"split_seed": s, "seed": t}
            lp_l, lp_g = c[f"test_novel/lstm/seed{t}"], c.get(f"test_novel/gpt/seed{t}")
            if lp_g is not None:
                r["gpt_minus_lstm_bits"] = cluster_ci(lp_l - lp_g, c["test_novel/cluster"])
                gap.append(r["gpt_minus_lstm_bits"]["est"])
            sw = f"sweep/test_novel/lstm/seed{t}"
            if sw in c:
                r["residual_beyond_composition_k256"] = cluster_ci(
                    c[sw] - c["sweep/test_novel/ngram_plus_cache"], c["sweep/test_novel/cluster"])
                resid.append(r["residual_beyond_composition_k256"]["est"])
            r["lstm_gain_over_1gram"] = one - prot[s]["neural"]["test_novel"][f"lstm/seed{t}"]["bits"]
            gain1.append(r["lstm_gain_over_1gram"])
            out["runs"].append(r)
    out["contrasts"] = {"gpt_minus_lstm_bits_heldout": stats(gap),
                        "lstm_residual_beyond_composition_k256": stats(resid),
                        "lstm_gain_over_1gram_heldout": stats(gain1),
                        "lstm_gain_over_1gram_pct_of_1gram": stats(
                            [g / prot[r["split_seed"]]["ngram"]["test_novel"]["1-gram"]["bits"] * 100
                             for g, r in zip(gain1, out["runs"])])}
    out["audit"] = {f"split{s}": prot[s]["audit"] for s in prot if prot[s]["audit"]}
    return out


def codon_bits(lp: np.ndarray, labels: np.ndarray) -> list[float]:
    return [float(-lp[labels == c].mean()) for c in range(len(CODON))]


def dna_intervals(dna: dict) -> dict:
    ng, nn = dna["ngram"]["test"], dna["neural"]["test"]
    names = ["uniform"] + [f"{o}-gram" for o in DNA_ORDERS]
    best = min((n for n in names if n != "uniform"), key=lambda n: ng[n]["bits"])
    lad = {n: stats([ng[n]["bits"]]) for n in names}
    lad["best n-gram"] = stats([ng[best]["bits"]])
    for m in ("gpt", "lstm"):
        lad[m] = stats([nn[f"{m}/seed{t}"]["bits"] for t in seeds_of(nn, m)])
    c = npz("dna")
    _, _, labels, _ = dna_targets()
    clu = c["test/cluster"]
    codon = {f"best n-gram ({best})": dict(zip(CODON, [stats([v]) for v in codon_bits(c[f"test/{best}"], labels)]))}
    gains, runs = {m: [] for m in ("gpt", "lstm")}, []
    for m in ("gpt", "lstm"):
        per_seed = [codon_bits(c[f"test/{m}/seed{t}"], labels) for t in seeds_of(nn, m)]
        codon[m] = {cl: stats([b[i] for b in per_seed]) for i, cl in enumerate(CODON)}
        for t in seeds_of(nn, m):
            d = c[f"test/{m}/seed{t}"] - c[f"test/{best}"]
            g = {cl: cluster_ci(d[labels == i], clu[labels == i]) for i, cl in enumerate(CODON)}
            gains[m].append(g)
            runs.append({"model": m, "seed": t, f"gain_over_{best}": g})
    gain = {m: {cl: stats([g[cl]["est"] for g in gains[m]]) for cl in CODON} for m in gains}
    comp = {"ngram_alone": stats([dna["cache"]["test"]["ngram_alone"][K]]),
            "ngram_plus_cache": stats([dna["cache"]["test"]["ngram_plus_cache"][K]])}
    for m in ("gpt", "lstm"):
        comp[m] = stats([dna["context"]["test"][f"{m}/seed{t}"][K] for t in seeds_of(dna["context"]["test"], m)])
    return {"best_ngram": best, "ladder": {"test": lad}, "codon_bits": codon, "codon_gain_over_best_ngram": gain,
            "composition": {"test": comp}, "runs": runs}


# --------------------------------------------------------------- figure 1
def fig_ladder(P, D):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    names = ["uniform"] + [f"{o}-gram" for o in range(1, 7)] + ["lstm", "gpt"]
    labels = ["uniform"] + [f"{o}-gram" for o in range(1, 7)] + ["LSTM", "GPT"]
    y = np.arange(len(names))[::-1]
    for split, col, lab in (("test_seen", INK2, "families seen in training"), ("test_novel", GPT, "held-out families")):
        lad = P["ladder"][split]
        xs, err = [lad[n]["mean"] for n in names], [lad[n]["sd"] for n in names]
        ax.errorbar(xs, y, xerr=err, fmt="none", ecolor=col, elinewidth=1.4, capsize=3, zorder=2)
        ax.scatter(xs, y, s=46, facecolor=SURF if split == "test_seen" else col,
                   edgecolor=col, linewidth=1.8, zorder=3, label=lab)
    for yi, n in zip(y, names):
        ax.plot([P["ladder"]["test_seen"][n]["mean"], P["ladder"]["test_novel"][n]["mean"]], [yi, yi],
                color=GRID, lw=2, zorder=1)
    ax.set_yticks(y, labels)
    ax.set_xlabel("bits per residue (lower is better); bars: ±1 sd over runs")
    ax.set_title("Protein: next-residue loss")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    names = ["uniform"] + [f"{o}-gram" for o in DNA_ORDERS] + ["lstm", "gpt"]
    labels = ["uniform"] + [f"{o}-gram" for o in DNA_ORDERS] + ["LSTM", "GPT"]
    y = np.arange(len(names))[::-1]
    lad = D["ladder"]["test"]
    xs, err = [lad[n]["mean"] for n in names], [lad[n]["sd"] for n in names]
    ax.errorbar(xs, y, xerr=err, fmt="none", ecolor=GPT, elinewidth=1.4, capsize=3, zorder=2)
    ax.scatter(xs, y, s=46, color=GPT, zorder=3)
    for yi, x in zip(y, xs):
        ax.text(x + 0.004, yi, f"{x:.3f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, labels)
    ax.set_xlabel("bits per nucleotide (lower is better)")
    ax.set_title("E. coli DNA: next-nucleotide loss (held-out blocks)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_ladder.png", dpi=160)
    plt.close(fig)


# --------------------------------------------------------------- figure 2
def fig_context(prot, dna):
    groups = {"protein": [prot[s] for s in prot], "dna": [dna]}
    panels = [("test_seen", "protein", "Protein, seen families"), ("test_novel", "protein", "Protein, held-out families"),
              ("test", "dna", "E. coli DNA, held-out blocks")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (split, kind, title) in zip(axes, panels):
        G = groups[kind]
        for name, col, lab in (("gpt", GPT, "GPT"), ("lstm", LSTM, "LSTM")):
            runs = [g["context"][split][f"{name}/seed{t}"] for g in G for t in seeds_of(g["context"][split], name)]
            ks = sorted(int(k) for k in runs[0])
            v = np.array([[r[str(k)] for k in ks] for r in runs])
            ax.plot(ks, v.mean(0), color=col, marker="o", ms=4, label=f"{lab} (mean of {len(runs)})")
            if len(runs) > 1:
                sd = v.std(0, ddof=1)
                ax.fill_between(ks, v.mean(0) - sd, v.mean(0) + sd, color=col, alpha=0.18, lw=0)
        ks = sorted(int(k) for k in G[0]["context"][split]["ngram"] if int(k) > 0)
        v = np.array([[g["context"][split]["ngram"][str(k)] for k in ks] for g in G])
        ax.plot(ks, v.mean(0), color=NGRAM, marker="s", ms=4, label="Kneser-Ney n-gram")
        ks = sorted(int(k) for k in G[0]["cache"][split]["ngram_plus_cache"])
        v = np.array([[g["cache"][split]["ngram_plus_cache"][str(k)] for k in ks] for g in G])
        ax.plot(ks, v.mean(0), color=NGRAM, ls="--", marker="s", ms=4, mfc=SURF,
                label=f"{G[0]['cache']['base_order']}-gram + own-composition cache")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("symbols of context given to the model")
        ax.set_title(title)
    axes[0].set_ylabel("bits per symbol; band: ±1 sd over runs")
    axes[1].set_ylim(4.04, 4.20)  # the high-order n-gram leaves this range; see fig 1
    axes[2].set_ylim(1.86, 2.01)
    axes[1].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_context.png", dpi=160)
    plt.close(fig)


# --------------------------------------------------------------- figure 3
def fig_codon(D):
    series = [(f"best n-gram ({D['best_ngram']})", NGRAM, f"best n-gram ({D['best_ngram']})"),
              ("lstm", LSTM, "LSTM"), ("gpt", GPT, "GPT")]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    w = 0.26
    for i, (key, col, lab) in enumerate(series):
        st = D["codon_bits"][key]
        x = np.arange(len(CODON)) + (i - 1) * (w + 0.02)
        ax.bar(x, [st[c]["mean"] for c in CODON], width=w, color=col, label=lab,
               yerr=[st[c]["sd"] for c in CODON], error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 2})
    ax.set_xticks(np.arange(len(CODON)), CODON)
    ax.set_ylim(1.5, 2.05)
    ax.set_ylabel("bits per nucleotide")
    ax.set_title("Where in the genome is the next base predictable?")
    ax.legend(ncol=3, fontsize=9, loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_codon_phase.png", dpi=160)
    plt.close(fig)


# --------------------------------------------------------------- figure 4
def fig_position_residue(prot):
    """Per-position and per-residue loss, averaged over every split and seed."""
    pos_curves = {}   # (split, key) -> list of per-run curves
    res_bits = {}     # key -> list of per-run 20-vectors (held-out, start residue excluded)
    pos0 = {}
    for s in prot:
        data.set_split_seed(s)
        c = npz(f"protein_split{s}")
        for split in ("test_seen", "test_novel"):
            flat, tg, _, pos, _ = protein_targets(split)
            keys = [("gpt", f"gpt/seed{t}") for t in seeds_of(prot[s]["neural"][split], "gpt")] + [("3-gram", "3-gram")]
            res = np.where(pos > 0, flat[tg], -1)
            for key, ck in keys:
                lp = c[f"{split}/{ck}"]
                pos_curves.setdefault((split, key), []).append([float(-lp[pos == p].mean()) for p in range(1, 60)])
                pos0.setdefault(f"{split}/{key}", []).append(float(-lp[pos == 0].mean()))
            if split == "test_novel":
                for key, ck in keys[:-1] + [("1-gram", "1-gram")]:
                    lp = c[f"{split}/{ck}"]
                    res_bits.setdefault(key, []).append([float(-lp[res == a].mean()) for a in range(20)])
    data.set_split_seed(0)
    out = {"position0_bits": {k: stats(v) for k, v in pos0.items()}}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    for split, ls in (("test_seen", "--"), ("test_novel", "-")):
        for key, col, lab in (("gpt", GPT, "GPT"), ("3-gram", NGRAM, "3-gram")):
            ax.plot(range(1, 60), np.mean(pos_curves[(split, key)], 0), color=col, ls=ls, lw=1.6,
                    label=f"{lab}, {'seen' if split == 'test_seen' else 'held-out'}")
    ax.set_xlabel("residue position (0 = first residue, omitted: see text)")
    ax.set_ylabel("bits per residue")
    ax.set_title("Loss along the N-terminus, after the start residue")
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1]
    mean = {k: np.mean(v, 0) for k, v in res_bits.items()}
    order = np.argsort(-mean["gpt"])
    for key, col, lab, dx in (("1-gram", MUTED, "1-gram (composition only)", -0.18), ("gpt", GPT, "GPT", 0.18)):
        sd = np.std(res_bits[key], 0, ddof=1) if len(res_bits[key]) > 1 else None
        ax.bar(np.arange(20) + dx, mean[key][order], width=0.34, color=col, label=lab,
               yerr=None if sd is None else sd[order], error_kw={"ecolor": INK2, "elinewidth": 0.8, "capsize": 1.5})
        out[key] = {AMINO[a]: stats([r[a] for r in res_bits[key]]) for a in order}
    ax.set_xticks(np.arange(20), [AMINO[a] for a in order])
    ax.set_ylabel("bits when this residue is next")
    ax.set_title("Held-out families: bits by next residue")
    ax.legend(fontsize=8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_position_residue.png", dpi=160)
    plt.close(fig)
    return out


# --------------------------------------------------------------- summary
def md_table(rows, header):
    s = "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n"
    return s + "".join("| " + " | ".join(r) + " |\n" for r in rows)


def summary_md(P, D, residues, prot, dna) -> str:
    L = ["# Results summary (generated by `python -m nexttok.report`)\n",
         f"Protein: {len(P['splits'])} family splits; neural cells pool split x seed "
         f"(n = {P['ladder']['test_novel']['lstm']['n']}). DNA: neural cells pool "
         f"{D['ladder']['test']['lstm']['n']} seeds. Cells are mean ± sample sd across runs.\n"]
    rows = []
    for n in P["ladder"]["test_seen"]:
        a, b = P["ladder"]["test_seen"][n], P["ladder"]["test_novel"][n]
        aa, ba = P["accuracy"]["test_seen"][n], P["accuracy"]["test_novel"][n]
        rows.append([n, pm(a), pm(aa), pm(b), pm(ba)])
    L.append("## Protein next-residue prediction\n")
    L.append(md_table(rows, ["model", "seen bits", "seen acc", "held-out bits", "held-out acc"]))
    L.append(f"\nBest n-gram order per split: {json.dumps(P['best_ngram_order'])}\n")
    L.append("\n## E. coli next-nucleotide prediction\n")
    L.append(md_table([[n, pm(v, 4)] for n, v in D["ladder"]["test"].items()], ["model", "bits"]))
    L.append(f"\n## DNA loss by genomic class (bits/nt, best n-gram = {D['best_ngram']})\n")
    L.append(md_table([[m] + [pm(D["codon_bits"][m][c]) for c in CODON] for m in D["codon_bits"]], ["model"] + CODON))
    L.append("\n**Gain over the best n-gram (bits/nt)**\n")
    L.append(md_table([[m] + [pm(D["codon_gain_over_best_ngram"][m][c]) for c in CODON]
                       for m in D["codon_gain_over_best_ngram"]], ["model"] + CODON))
    L.append("\nPer-seed 95% bootstrap intervals over 5 kb chunks:\n")
    rows = [[r["model"], str(r["seed"])] + [f"{g['est']:.3f} [{g['lo']:.3f}, {g['hi']:.3f}]"
                                            for g in r[f"gain_over_{D['best_ngram']}"].values()] for r in D["runs"]]
    L.append(md_table(rows, ["model", "seed"] + CODON))
    L.append(f"\n## Composition control at k = {K}\n")
    for kind, X in (("protein", P), ("dna", D)):
        for split, comp in X["composition"].items():
            L.append(f"\n**{kind}, {split}**\n")
            L.append(md_table([[k, pm(v, 4 if "fraction" not in k else 3)] for k, v in comp.items()], ["quantity", "value"]))
    L.append("\n## Paired contrasts on held-out families, per run (95% family bootstrap)\n")
    rows = []
    for r in P["runs"]:
        g, c = r.get("gpt_minus_lstm_bits"), r.get("residual_beyond_composition_k256")
        rows.append([str(r["split_seed"]), str(r["seed"]),
                     f"{g['est']:.4f} [{g['lo']:.4f}, {g['hi']:.4f}]" if g else "–",
                     f"{c['est']:.4f} [{c['lo']:.4f}, {c['hi']:.4f}]" if c else "–",
                     f"{r['lstm_gain_over_1gram']:.4f}"])
    L.append(md_table(rows, ["split", "seed", "GPT − LSTM bits", f"residual beyond composition (k={K})",
                             "LSTM gain over 1-gram"]))
    L.append("\n" + md_table([[k, pm(v, 4)] for k, v in P["contrasts"].items()], ["contrast", "mean ± sd"]))
    for kind, groups in (("protein", [(f"split{s}", prot[s]) for s in prot]), ("dna", [("", dna)])):
        for tag, g in groups:
            c = g["context"]
            if not c:
                continue
            L.append(f"\n## Truncated-context sweep: {kind} {tag}\n")
            for split, r in c.items():
                ks = sorted(int(k) for k in next(v for m, v in r.items() if m != "ngram"))
                rows = [[m] + [f"{r[m][str(k)]:.3f}" for k in ks] for m in sorted(r) if m != "ngram"]
                rows.append(["n-gram"] + [f"{r['ngram'][str(k)]:.3f}" if str(k) in r["ngram"] else "–" for k in ks])
                L.append(f"\n**{split}** (context length →)\n\n" + md_table(rows, ["model"] + [str(k) for k in ks]))
    for s in prot:
        for name in ("audit", "generate"):
            c = prot[s][name]
            if c:
                c = json.loads(json.dumps(c))
                for v in c.values():
                    if isinstance(v, dict):
                        v.pop("composition", None)
                L.append(f"\n## {name}, split {s}\n\n```json\n{json.dumps(c, indent=2)}\n```\n")
    L.append("\n## Per-residue bits, held-out families (mean over runs)\n\n```json\n"
             + json.dumps({k: {a: round(v["mean"], 3) for a, v in d.items()} for k, d in residues.items()},
                          indent=1) + "\n```\n")
    return "\n".join(L)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    prot, dna = load_groups()
    P, D = protein_intervals(prot), dna_intervals(dna)
    fig_ladder(P, D)
    fig_context(prot, dna)
    fig_codon(D)
    residues = fig_position_residue(prot)
    (RES / "intervals.json").write_text(json.dumps({"protein": P, "dna": D, "residues": residues}, indent=1))
    (RES / "summary.md").write_text(summary_md(P, D, residues, prot, dna))
    print((RES / "summary.md").read_text()[:6000])


if __name__ == "__main__":
    main()
