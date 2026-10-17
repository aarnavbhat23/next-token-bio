"""Turn results/*.json and results/cache/*.npz into figures and summary.md."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .evaluate import CACHE, DNA_ORDERS, dna_targets, protein_targets
from .prepare import DNA_LABELS
from .vocab import AMINO

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = RES / "figures"

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


def load(name):
    p = RES / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def best_ngram(d: dict) -> tuple[str, dict]:
    k = min((k for k in d if k.endswith("-gram")), key=lambda k: d[k]["bits"])
    return k, d[k]


# --------------------------------------------------------------- figure 1
def fig_ladder(prot, dna):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    names = ["uniform"] + [f"{o}-gram" for o in range(1, 7)] + ["lstm", "gpt"]
    labels = ["uniform"] + [f"{o}-gram" for o in range(1, 7)] + ["LSTM", "GPT"]
    y = np.arange(len(names))[::-1]
    for split, mk, col, lab in (("test_seen", "o", INK2, "families seen in training"),
                                ("test_novel", "o", GPT, "held-out families")):
        xs = [prot[split][n]["bits"] for n in names]
        ax.scatter(xs, y, s=46, marker=mk, facecolor=SURF if split == "test_seen" else col,
                   edgecolor=col, linewidth=1.8, zorder=3, label=lab)
    for yi, n in zip(y, names):
        a, b = prot["test_seen"][n]["bits"], prot["test_novel"][n]["bits"]
        ax.plot([a, b], [yi, yi], color=GRID, lw=2, zorder=1)
    ax.set_yticks(y, labels)
    ax.set_xlabel("bits per residue (lower is better)")
    ax.set_title("Protein: next-residue loss")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    names = ["uniform"] + [f"{o}-gram" for o in DNA_ORDERS] + ["lstm", "gpt"]
    labels = ["uniform"] + [f"{o}-gram" for o in DNA_ORDERS] + ["LSTM", "GPT"]
    y = np.arange(len(names))[::-1]
    xs = [dna["test"][n]["bits"] for n in names]
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
def fig_context(cp, cd, kp, kd):
    panels = [("test_seen", cp, kp, "Protein, seen families"), ("test_novel", cp, kp, "Protein, held-out families"),
              ("test", cd, kd, "E. coli DNA, held-out blocks")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (split, src, cache, title) in zip(axes, panels):
        r = src[split]
        for name, col, lab in (("gpt", GPT, "GPT"), ("lstm", LSTM, "LSTM")):
            ks = sorted(int(k) for k in r[name])
            ax.plot(ks, [r[name][str(k)] for k in ks], color=col, marker="o", ms=4, label=lab)
        ks = sorted(int(k) for k in r["ngram"] if int(k) > 0)
        ax.plot(ks, [r["ngram"][str(k)] for k in ks], color=NGRAM, marker="s", ms=4, label="Kneser-Ney n-gram")
        c = cache[split]["ngram_plus_cache"]
        ks = sorted(int(k) for k in c)
        ax.plot(ks, [c[str(k)] for k in ks], color=NGRAM, ls="--", marker="s", ms=4, mfc=SURF,
                label=f"{cache['base_order']}-gram + own-composition cache")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("symbols of context given to the model")
        ax.set_title(title)
    axes[0].set_ylabel("bits per symbol")
    axes[1].set_ylim(4.06, 4.20)  # the high-order n-gram leaves this range; see fig 1
    axes[2].set_ylim(1.87, 2.01)
    axes[1].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_context.png", dpi=160)
    plt.close(fig)


# --------------------------------------------------------------- figure 3
def fig_codon(dna):
    cache = dict(np.load(CACHE / "dna_logp.npz"))
    _, _, labels, _ = dna_targets()
    bk, _ = best_ngram(dna["test"])
    series = [(f"test/{bk}", NGRAM, f"best n-gram ({bk})"), ("test/lstm", LSTM, "LSTM"), ("test/gpt", GPT, "GPT")]
    classes = [0, 1, 2, 3, 4, 5, 6]
    names = ["non-coding", "+ codon 1", "+ codon 2", "+ codon 3", "− codon 1", "− codon 2", "− codon 3"]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    w = 0.26
    table = {}
    for i, (key, col, lab) in enumerate(series):
        lp = cache[key]
        vals = [float(-lp[labels == c].mean()) for c in classes]
        table[lab] = dict(zip(names, vals))
        ax.bar(np.arange(len(classes)) + (i - 1) * (w + 0.02), vals, width=w, color=col, label=lab)
    ax.set_xticks(np.arange(len(classes)), names)
    ax.set_ylim(1.5, 2.05)
    ax.set_ylabel("bits per nucleotide")
    ax.set_title("Where in the genome is the next base predictable?")
    ax.legend(ncol=3, fontsize=9, loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_codon_phase.png", dpi=160)
    plt.close(fig)
    return table


# --------------------------------------------------------------- figure 4
def fig_position_residue(prot):
    cache = dict(np.load(CACHE / "protein_logp.npz"))
    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    for split, ls in (("test_seen", "--"), ("test_novel", "-")):
        flat, tg, _, pos, _ = protein_targets(split)
        for key, col, lab in (("gpt", GPT, "GPT"), ("3-gram", NGRAM, "3-gram")):
            lp = cache[f"{split}/{key}"]
            m = [float(-lp[pos == p].mean()) for p in range(1, 60)]
            out.setdefault("position0_bits", {})[f"{split}/{key}"] = float(-lp[pos == 0].mean())
            ax.plot(range(1, 60), m, color=col, ls=ls, lw=1.6,
                    label=f"{lab}, {'seen' if split == 'test_seen' else 'held-out'}")
    ax.set_xlabel("residue position (0 = first residue, omitted: see text)")
    ax.set_ylabel("bits per residue")
    ax.set_title("Loss along the N-terminus, after the start residue")
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1]
    flat, tg, _, pos, _ = protein_targets("test_novel")
    res = np.where(pos > 0, flat[tg], -1)  # exclude the start residue
    order = np.argsort([-cache["test_novel/gpt"][res == a].mean() for a in range(20)])
    for key, col, lab, dx in (("1-gram", MUTED, "1-gram (composition only)", -0.18), ("gpt", GPT, "GPT", 0.18)):
        lp = cache[f"test_novel/{key}"]
        vals = [float(-lp[res == a].mean()) for a in order]
        out[key] = {AMINO[a]: v for a, v in zip(order, vals)}
        ax.bar(np.arange(20) + dx, vals, width=0.34, color=col, label=lab)
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


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    ng_p, nn_p = load("ngram_protein"), load("neural_protein")
    ng_d, nn_d = load("ngram_dna"), load("neural_dna")
    prot = {s: {**ng_p[s], **nn_p[s]} for s in ng_p}
    dna = {"test": {**ng_d["test"], **nn_d["test"]}}
    fig_ladder(prot, dna)
    cp, cd = load("context_protein"), load("context_dna")
    kp, kd = load("cache_protein"), load("cache_dna")
    if cp and cd and kp and kd:
        fig_context(cp, cd, kp, kd)
    codon = fig_codon(dna)
    residues = fig_position_residue(prot)

    lines = ["# Results summary (generated by `python -m nexttok.report`)\n"]
    rows = []
    for n in prot["test_seen"]:
        a, b = prot["test_seen"][n], prot["test_novel"][n]
        rows.append([n, f"{a['bits']:.3f}", f"{a['perplexity']:.2f}", f"{a['accuracy']:.3f}",
                     f"{b['bits']:.3f}", f"{b['perplexity']:.2f}", f"{b['accuracy']:.3f}"])
    lines.append("## Protein next-residue prediction\n")
    lines.append(md_table(rows, ["model", "seen bits", "seen ppl", "seen acc", "held-out bits", "held-out ppl", "held-out acc"]))
    rows = [[n, f"{v['bits']:.4f}", f"{v['perplexity']:.3f}", f"{v['accuracy']:.3f}"] for n, v in dna["test"].items()]
    lines.append("\n## E. coli next-nucleotide prediction\n")
    lines.append(md_table(rows, ["model", "bits", "ppl", "acc"]))
    lines.append("\n## DNA loss by genomic class (bits/nt)\n")
    cls = list(next(iter(codon.values())).keys())
    lines.append(md_table([[m] + [f"{codon[m][c]:.3f}" for c in cls] for m in codon], ["model"] + cls))
    for name in ("context_protein", "context_dna"):
        c = load(name)
        if not c:
            continue
        lines.append(f"\n## Truncated-context sweep: {name}\n")
        for split, r in c.items():
            ks = sorted(int(k) for k in r["gpt"])
            rows = [[m] + [f"{r[m][str(k)]:.3f}" if str(k) in r[m] else "" for k in ks] for m in ("gpt", "lstm")]
            rows.append(["n-gram"] + [f"{r['ngram'][str(k)]:.3f}" if str(k) in r["ngram"] else "–" for k in ks])
            lines.append(f"\n**{split}** (context length →)\n\n" + md_table(rows, ["model"] + [str(k) for k in ks]))
    for name in ("cache_protein", "cache_dna"):
        c = load(name)
        if not c:
            continue
        lines.append(f"\n## Composition control: {name} (base order {c['base_order']}, tuned on validation)\n")
        for split in [k for k in c if k.startswith("test")]:
            ks = sorted(int(k) for k in c[split]["ngram_plus_cache"])
            rows = [[m] + [f"{c[split][m][str(k)]:.3f}" for k in ks] for m in ("ngram_alone", "ngram_plus_cache")]
            lines.append(f"\n**{split}**\n\n" + md_table(rows, ["model"] + [str(k) for k in ks]))
    for name in ("audit_protein", "generate_protein"):
        c = load(name)
        if c:
            for v in c.values():
                if isinstance(v, dict):
                    v.pop("composition", None)
            lines.append(f"\n## {name}\n\n```json\n{json.dumps(c, indent=2)}\n```\n")
    lines.append("\n## Per-residue bits, held-out families\n\n```json\n" + json.dumps(residues, indent=1) + "\n```\n")
    (RES / "summary.md").write_text("\n".join(lines))
    print((RES / "summary.md").read_text()[:4000])


if __name__ == "__main__":
    main()
