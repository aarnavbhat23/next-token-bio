"""Figures and tables for the reading-frame analysis (results/frame/*.json).

  python -m nexttok.frame_report     # results/figures/frame_*.png, results/frame/summary.md
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .frame import CLASS_NAMES, OUT

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"

# reference categorical palette; the first three match report.py's GPT / LSTM / n-gram.
# Bar order (oracle, HMM, LSTM, GPT) and line order (n-gram, LSTM, GPT, oracle, HMM,
# window) were each run through the dataviz palette validator: all checks pass.
GPT, LSTM, NGRAM, ORACLE, HMM, WINDOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#e87ba4"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 2,
})

GENOME_ORDER = ["ecoli", "bsubtilis", "scoelicolor"]
SHORT = {"ecoli": "E. coli K-12", "bsubtilis": "B. subtilis 168", "scoelicolor": "S. coelicolor A3(2)"}
BAR_SERIES = [("frame_oracle_o5", ORACLE, "frame Markov, true frame (oracle)"),
              ("frame_hmm", HMM, "frame Markov, frame inferred (HMM)"),
              ("lstm", LSTM, "LSTM"), ("gpt", GPT, "GPT")]


def load(name: str):
    p = OUT / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def saved(r: dict, model: str, cls: str) -> float:
    """Bits per base saved relative to the genome's best n-gram (positive = better)."""
    return r["test"][r["best_ngram"]][cls] - r["test"][model][cls]


# --------------------------------------------------------------- figures
SHORT_CLASS = ["non-\ncoding", "+1", "+2", "+3", "−1", "−2", "−3"]


def gain_bars(ax, r: dict, classes=CLASS_NAMES, legend=True, short=False):
    x = np.arange(len(classes))
    w = 0.2
    for i, (key, col, lab) in enumerate(BAR_SERIES):
        vals = [saved(r, key, c) for c in classes]
        ax.bar(x + (i - 1.5) * (w + 0.015), vals, width=w, color=col, label=lab)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(x, SHORT_CLASS if short else classes)
    ax.grid(axis="x", visible=False)
    if legend:
        ax.legend(fontsize=8, loc="upper left")


def fig_classes(r: dict):
    fig, ax = plt.subplots(figsize=(9.5, 4))
    gain_bars(ax, r)
    ax.set_ylabel(f"bits per base saved vs {r['best_ngram']}")
    ax.set_title("E. coli: does knowing the reading frame reproduce the LSTM's gain?")
    fig.tight_layout()
    fig.savefig(FIG / "frame_classes.png", dpi=160)
    plt.close(fig)


def fig_context(r: dict):
    s = r["sweep"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    series = [("ngram", NGRAM, "-", "s", "Kneser-Ney n-gram"), ("lstm", LSTM, "-", "o", "LSTM"),
              ("gpt", GPT, "-", "o", "GPT"), ("frame_oracle", ORACLE, "-", "D", "frame Markov, true frame"),
              ("frame_hmm", HMM, "-", "^", "frame Markov, HMM-inferred frame"),
              ("frame_window", WINDOW, "--", "v", "frame Markov, window-inferred frame")]
    for ax, src, title in ((axes[0], s, f"All {s['n']:,} targets"),
                           (axes[1], s["codon_read_last"], f"Last base read in each codon (n = {s['n_codon_read_last']:,})")):
        for key, col, ls, mk, lab in series:
            d = src[key]
            ks = sorted(int(k) for k in d if int(k) > 0)
            ax.plot(ks, [d[str(k)] for k in ks], color=col, ls=ls, marker=mk, ms=4.5, lw=1.8, label=lab,
                    mfc=SURF if ls == "--" else col)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("bases of context given to the model")
        ax.set_title(title)
    axes[0].set_ylabel("bits per base")
    axes[0].set_ylim(1.855, 2.005)  # the 10- and 12-gram leave this range; see fig 1
    axes[1].set_ylim(1.755, 2.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIG / "frame_context.png", dpi=160)
    plt.close(fig)


def probe_rows(p: dict) -> list[tuple[str, float]]:
    """Every trained layer; the best layer of each untrained network; references. Ascending."""
    rows = [("majority class", p["majority_accuracy"])]
    best_untrained: dict[str, tuple[str, float]] = {}
    for k, v in p["probes"].items():
        if "untrained" in k:
            arch = k.split()[0]
            if v["accuracy"] > best_untrained.get(arch, ("", -1))[1]:
                best_untrained[arch] = (f"{arch}, untrained (best layer)", v["accuracy"])
        else:
            rows.append((k, v["accuracy"]))
    rows += list(best_untrained.values())
    if "frame_hmm_argmax" in p:
        rows.append(("frame Markov HMM posterior", p["frame_hmm_argmax"]["accuracy"]))
    return sorted(rows, key=lambda r: r[1])


def fig_probe(p: dict):
    rows = probe_rows(p)

    def colour(name):
        if name.startswith("frame"):
            return HMM
        if "untrained" in name or name.startswith("8-gram") or name == "majority class":
            return MUTED
        return LSTM if name.startswith("LSTM") else GPT

    fig, ax = plt.subplots(figsize=(8, 0.34 * len(rows) + 1.3))
    y = np.arange(len(rows))
    ax.barh(y, [v for _, v in rows], color=[colour(n) for n, _ in rows], height=0.7)
    for yi, (_, v) in zip(y, rows):
        ax.text(v + 0.008, yi, f"{v:.3f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, [n for n, _ in rows], fontsize=8.5)
    ax.set_xlabel("7-class probe accuracy, held-out blocks (non-coding, ± codon 1-3)")
    ax.set_title("E. coli: linear probes for the reading frame")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG / "frame_probe.png", dpi=160)
    plt.close(fig)


def fig_genomes(rs: list[dict]):
    fig, axes = plt.subplots(1, len(rs), figsize=(4.6 * len(rs), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, rs):
        gain_bars(ax, r, legend=False, short=True)
        ax.set_title(f"{SHORT[r['genome']]} (vs {r['best_ngram']})")
        ax.set_xlabel("class: strand and codon position")
    axes[0].set_ylabel("bits per base saved vs best n-gram")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "frame_genomes.png", dpi=160)
    plt.close(fig)


# --------------------------------------------------------------- tables
def md_table(rows, header):
    s = "| " + " | ".join(header) + " |\n|" + "---|" + "---:|" * (len(header) - 1) + "\n"
    return s + "".join("| " + " | ".join(r) + " |\n" for r in rows)


NAMES = {"lstm": "LSTM", "gpt": "GPT", "frame_hmm": "frame Markov o5, HMM-inferred frame",
         "frame_window_k256": "frame Markov o5, window-inferred frame (k = 256)"}


def model_label(key: str) -> str:
    if key.startswith("frame_oracle_o"):
        return f"frame Markov o{key[len('frame_oracle_o'):]}, true frame"
    return NAMES.get(key, key)


def class_table(r: dict, keys: list[str]) -> str:
    cols = CLASS_NAMES + ["all"]
    rows = [[model_label(k)] + [f"{r['test'][k][c]:.3f}" for c in cols] for k in keys]
    return md_table(rows, ["bits/nt"] + cols)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    rs = [r for r in (load(g) for g in GENOME_ORDER) if r]
    lines = ["# Reading-frame analysis (generated by `python -m nexttok.frame_report`)\n"]
    keys = None
    for r in rs:
        keys = [r["best_ngram"], "frame_oracle_o5", "frame_oracle_o7", "frame_window_k256", "frame_hmm", "lstm", "gpt"]
        lines.append(f"\n## {r['species']} — {r['n_test']:,} held-out targets\n")
        lines.append(class_table(r, keys))
        lines.append("\nOracle ladder (true frame, context m bases): "
                     + ", ".join(f"m={m}: {r['test'][f'frame_oracle_o{m}']['all']:.3f}" for m in range(8)) + "\n")
        lines.append("\nKneser-Ney ladder: " + ", ".join(f"{k}: {v:.3f}" for k, v in r["ngram_ladder"].items()) + "\n")
        ci = r["class_inference"]
        lines.append("\nClass inference accuracy (7 classes / frame on coding bases): "
                     + "; ".join(f"{k}: {v['accuracy_7class']:.3f} / {v['frame_accuracy_on_coding']:.3f}" for k, v in ci.items()) + "\n")
        s = r["sweep"]
        rows = [[k] + [f"{s[k][str(x)]:.3f}" for x in s["k"]]
                for k in ("lstm", "gpt", "frame_oracle", "frame_window", "frame_hmm")]
        lines.append(f"\n**Context sweep** ({s['n']:,} targets; bits by context k)\n\n" + md_table(rows, ["model"] + [str(x) for x in s["k"]]))
        c = s["codon_read_last"]
        rows = [[k] + [f"{c[k][str(x)]:.3f}" for x in s["k"]]
                for k in ("lstm", "gpt", "frame_oracle", "frame_window", "frame_hmm")]
        lines.append(f"\n**Context sweep, last base read in each codon** (n = {s['n_codon_read_last']:,})\n\n"
                     + md_table(rows, ["model"] + [str(x) for x in s["k"]]))
        ng = s["ngram"]
        lines.append("\nn-gram on the sweep targets, by context length: " + ", ".join(f"{k}: {v:.3f}" for k, v in ng.items()) + "\n")
    ec = next((r for r in rs if r["genome"] == "ecoli"), None)
    if ec:
        fig_classes(ec)
        fig_context(ec)
    if len(rs) > 1:
        fig_genomes(rs)
    for g in GENOME_ORDER:
        p = load(f"probe_{g}")
        if not p:
            continue
        if g == "ecoli":
            fig_probe(p)
        rows = [[k, f"{v['accuracy']:.3f}", f"{v['frame_accuracy_on_coding']:.3f}", f"{v['class_info_bits']:.3f}"]
                for k, v in p["probes"].items()]
        rows.append(["majority class", f"{p['majority_accuracy']:.3f}", "", "0"])
        if "frame_hmm_argmax" in p:
            h = p["frame_hmm_argmax"]
            rows.append(["frame HMM posterior (argmax)", f"{h['accuracy']:.3f}", f"{h['frame_accuracy_on_coding']:.3f}", ""])
        lines.append(f"\n## Linear probes, {SHORT[g]} ({p['n_train']:,} validation / {p['n_test']:,} test positions)\n\n"
                     + md_table(rows, ["features", "7-class accuracy", "accuracy on coding bases", "class info (bits)"]))
        if "lstm_gain_by_probe" in p:
            lines.append("\n```json\n" + json.dumps(p["lstm_gain_by_probe"], indent=1) + "\n```\n")
    (OUT / "summary.md").write_text("\n".join(lines))
    print((OUT / "summary.md").read_text()[:6000])


if __name__ == "__main__":
    main()
