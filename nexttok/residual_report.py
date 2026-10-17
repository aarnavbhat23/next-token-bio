"""Figures 5-6 and results/residual_summary.md from the residual, homology and variant JSONs.

  python -m nexttok.residual_report
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from .report import FIG, GPT, GRID, INK2, LSTM, MUTED, NGRAM, RES, SURF, md_table

PERIODIC = "#8a5cc2"  # a fourth categorical slot for the explicit residual baselines

LADDER = [("3-gram", "3-gram"), ("cache", "+ own composition, last 256 (README)"),
          ("cache+periodic (linear mix)", "+ hydrophobic lags 2,3,4,7, linear mix"),
          ("cache+periodic", "+ hydrophobic lags 2,3,4,7, log-linear"),
          ("cache+periodic (3 classes)", "+ 3-class lags 2,3,4,7, log-linear"),
          ("local", "+ local composition, last 8-256"),
          ("local+periodic", "+ local composition + hydrophobic lags"),
          ("local+periodic+lags 2-16", "+ local + hydrophobic lags + identity lags 2-16")]


def load(name):
    p = RES / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def mb(x):
    return 1000 * x


# --------------------------------------------------------------- figure 5
def fig_residual(r):
    split = "test_novel"
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), gridspec_kw={"width_ratios": [1.35, 1, 1.3]})
    ax = axes[0]
    rows = [(lab, r["baselines"][k][split], NGRAM if k in ("3-gram", "cache") else PERIODIC) for k, lab in LADDER]
    rows += [("GPT", r["neural"]["gpt"][split], GPT), ("LSTM", r["neural"]["lstm"][split], LSTM)]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, v, col) in zip(y, rows):
        ax.scatter(v, yi, s=44, color=col, zorder=3)
        ax.text(v + 0.0015, yi, f"{v:.3f}", va="center", fontsize=8, color=INK2)
    for k, ls in (("cache", "--"),):
        ax.axvline(r["baselines"][k][split], color=NGRAM, ls=ls, lw=1)
    ax.axvline(r["neural"]["lstm"][split], color=LSTM, ls="--", lw=1)
    ax.set_yticks(y, [lab for lab, _, _ in rows], fontsize=8.5)
    ax.set_xlim(4.075, 4.16)
    ax.set_xlabel("bits per residue at k = 256 (lower is better)")
    ax.set_title("Held-out families: explicit baselines vs neural")
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    scan = r["lag_scan_gain_over_cache"]
    lags = sorted(int(k) for k in scan)
    for L in (3, 4, 7, 11, 14):
        ax.axvspan(L - 0.45, L + 0.45, color=GRID, zorder=0)
    ax.plot(lags, [mb(scan[str(L)]["hydrophobic class"][split]) for L in lags], color=PERIODIC, marker="o", ms=4,
            label="hydrophobic / not at lag L")
    ax.plot(lags, [mb(scan[str(L)]["identity"][split]) for L in lags], color=MUTED, marker="s", ms=4,
            label="residue identity at lag L")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(lags, [str(L) for L in lags], fontsize=8)
    ax.set_xlabel("lag L behind the target (shaded: helix face i-3, i-4, i-7, ...)")
    ax.set_ylabel("millibits gained over the composition cache")
    ax.set_title("One lag at a time, held-out families")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[2]
    pt = r["per_target_test_novel"]
    groups = [g for g in ("disorder (metapredict)", "moment", "hydropathy") if g in pt]
    short = {"disordered": "disord.", "weak periodicity": "weak", "strand-like": "strand", "helix-like": "helix",
             "soluble-like": "soluble", "TM-like (KD >= 1.6)": "TM-like"}
    titles = {"disorder (metapredict)": "disorder", "moment": "hydrophobic moment", "hydropathy": "hydropathy"}
    labels, pos, x = [], [], 0.0
    for g in groups:
        x0 = x
        for lab in sorted(pt[g], key=lambda k: -pt[g][k]["fraction"]):
            row = pt[g][lab]
            for key, col, dx in (("lstm_minus_cache", LSTM, -0.2), ("best_explicit_minus_cache", PERIODIC, 0.2)):
                lo, hi = row[f"{key}_ci95"]
                ax.bar(x + dx, mb(row[key]), width=0.38, color=col,
                       label={"lstm_minus_cache": "LSTM", "best_explicit_minus_cache": "best explicit baseline"}[key]
                       if not labels else None)
                ax.plot([x + dx] * 2, [mb(lo), mb(hi)], color=INK2, lw=1)
            labels.append(f"{short.get(lab, lab)}\n{100 * row['fraction']:.0f}%")
            pos.append(x)
            x += 1.2
        ax.text((x0 + x - 1.2) / 2, -0.17, titles[g], transform=ax.get_xaxis_transform(), ha="center",
                fontsize=8.5, color=INK2)
        x += 0.6
    ax.set_xticks(pos, labels, fontsize=7.5)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel("mean millibits gained over the cache model")
    ax.set_title("Gain over the cache model, by class")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_residual.png", dpi=160)
    plt.close(fig)


# --------------------------------------------------------------- figure 6
def fig_homology_variants(h, v):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.5), gridspec_kw={"width_ratios": [1, 1.55]})
    ax = axes[0]
    names = [("1-gram", "1-gram"), ("3-gram", "3-gram"), ("5-gram", "5-gram"), ("gpt", "GPT"), ("lstm", "LSTM")]
    F = h["filters"]
    strict = F.get("cluster_or_search", F["cluster"])
    series = [(F["cluster"], "kept", "o", GPT, f"kept, no 30% cluster mate ({F['cluster']['kept']})"),
              (strict, "kept", "s", NGRAM, f"kept, no 30% relative at all ({strict['kept']})"),
              (strict, "dropped", "x", LSTM, f"the {strict['dropped']} dropped proteins")]
    y = np.arange(len(names))[::-1]
    for i, (f, part, mk, col, lab) in enumerate(series):
        xs = [mb(f["bits"][k][part] - f["bits"][k]["all"]) for k, _ in names]
        ax.scatter(xs, y + 0.18 - 0.18 * i, s=40, marker=mk, color=col, zorder=3, label=lab)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y, [lab for _, lab in names])
    ax.set_xlabel("millibits per residue vs all 3,000 held-out proteins")
    ax.set_title("Dropping held-out proteins with a training relative")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    assays = list(v["assays"])
    series = [("1-gram", MUTED, "1-gram (composition)"), ("3-gram", NGRAM, "3-gram"), ("gpt", GPT, "GPT"),
              ("lstm", LSTM, "LSTM")]
    w = 0.17
    for i, (k, col, lab) in enumerate(series):
        for j, a in enumerate(assays):
            s = v["assays"][a]["spearman"][k]
            xx = j + (i - 2) * w
            ax.bar(xx, s["rho"], width=w * 0.92, color=col, label=lab if j == 0 else None)
            ax.plot([xx] * 2, s["ci95"], color=INK2, lw=1)
    for j, a in enumerate(assays):
        p = v["assays"][a].get("published_spearman", {})
        for m, mk in (("CARP (600K)", "o"), ("ESM2 (8M)", "D"), ("ESM2 (650M)", "^")):
            if m in p:
                ax.scatter(j + 2 * w, p[m], marker=mk, s=28, facecolor=SURF if mk == "o" else INK2,
                           edgecolor=INK2, zorder=4, label=f"{m}, published" if j == 0 else None)
    ax.axhline(0, color=INK2, lw=0.8)
    kind = {"OrganismalFitness": "fitness", "Activity": "activity", "Expression": "abundance",
            "Binding": "binding", "Stability": "stability"}
    short = [a.split("_")[0] + "\n" + kind.get(v["assays"][a]["selection_type"], "") for a in assays]
    ax.set_xticks(np.arange(len(assays)), short, fontsize=8.5)
    ax.set_ylim(-0.25, 0.95)
    ax.set_ylabel("Spearman ρ with measured fitness")
    ax.set_title("Zero-shot variant effects, six ProteinGym DMS assays")
    ax.legend(fontsize=7.5, ncol=4, loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_homology_variants.png", dpi=160)
    plt.close(fig)


# --------------------------------------------------------------- summary
def summary(r, h, v) -> str:
    out = ["# Residual, homology filter and variant effects (generated by `python -m nexttok.residual_report`)\n"]
    if r:
        rows = []
        extra = [("cache+periodic (weights fit on validation)", "+ hydrophobic lags, weights fit on validation")]
        for k, lab in LADDER + extra:
            b = r["baselines"][k]
            rows.append([lab, f"{b['test_novel']:.4f}", f"{int(round(100 * b['residual_closed']['test_novel']))}%",
                         f"{b['test_seen']:.4f}"])
        for k in ("gpt", "lstm"):
            n = r["neural"][k]
            rows.append([k.upper(), f"{n['test_novel']:.4f}", "", f"{n['test_seen']:.4f}"])
        out.append("## Explicit baselines at k = 256\n")
        out.append(md_table(rows, ["model", "held-out bits", "LSTM residual closed", "seen bits"]))
        scan = r["lag_scan_gain_over_cache"]
        rows = [[L, f"{mb(scan[L]['hydrophobic class']['test_novel']):.2f}",
                 f"{mb(scan[L]['identity']['test_novel']):.2f}"] for L in scan]
        out.append("\n## Single-lag gain over the cache model, held-out (millibits)\n")
        out.append(md_table(rows, ["lag", "hydrophobic class", "identity"]))
        for g, d in r["per_target_test_novel"].items():
            rows = [[lab, str(x["targets"]), f"{mb(x['lstm_minus_cache']):.1f}",
                     f"[{mb(x['lstm_minus_cache_ci95'][0]):.1f}, {mb(x['lstm_minus_cache_ci95'][1]):.1f}]",
                     f"{100 * x['lstm_minus_cache_share_of_total']:.0f}%", f"{mb(x['periodic_minus_cache']):.1f}",
                     f"{mb(x['best_explicit_minus_cache']):.1f}"] for lab, x in d.items()]
            out.append(f"\n## LSTM gain over the cache model by {g} (millibits per target)\n")
            out.append(md_table(rows, ["class", "targets", "LSTM", "95% CI", "share of LSTM gain",
                                       "periodic", "best explicit"]))
    if h:
        for fname, f in h["filters"].items():
            rows = [[k, f"{b['all']:.4f}", f"{b['kept']:.4f}", f"{b['dropped']:.4f}",
                     f"{b['kept'] - b['all']:+.4f}", f"{b['gain_over_1gram_all']:.4f}",
                     f"{b['gain_over_1gram_kept']:.4f}"] for k, b in f["bits"].items()]
            out.append(f"\n## Homology filter `{fname}`: dropped {f['dropped']} of "
                       f"{h['eval_subset']['sequences']} proteins\n")
            out.append(md_table(rows, ["model", "all", "kept", "dropped", "kept - all", "gain/1-gram all",
                                       "gain/1-gram kept"]))
    if v:
        names = list(v["mean_spearman"])
        rows = []
        for a, d in v["assays"].items():
            p = d.get("published_spearman", {})
            rows.append([a, d["selection_type"], str(d["mutants"])] +
                        [f"{d['spearman'][k]['rho']:.3f}" for k in names] +
                        [f"{p.get(m, float('nan')):.3f}" for m in v["reference_models"]])
        rows.append(["mean", "", ""] + [f"{v['mean_spearman'][k]:.3f}" for k in names] + [""] * len(v["reference_models"]))
        out.append("\n## Zero-shot Spearman rho (ProteinGym)\n")
        out.append(md_table(rows, ["assay", "selection", "mutants"] + names + list(v["reference_models"])))
    return "\n".join(out)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    r, h, v = load("residual_protein"), load("homology_protein"), load("variants_protein")
    if r:
        fig_residual(r)
    if h and v:
        fig_homology_variants(h, v)
    text = summary(r, h, v)
    (RES / "residual_summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
