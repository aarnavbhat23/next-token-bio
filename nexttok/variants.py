"""Does lower next-token loss help zero-shot variant-effect prediction?

Six small ProteinGym deep mutational scanning assays (Notin et al., NeurIPS 2023;
wild type <= 512 residues, one per selection type plus a second organismal-fitness
assay) are scored zero-shot. Every mutant gets the autoregressive log-likelihood
ratio over the full sequence,

    s(mutant) = log2 P(mutant sequence) - log2 P(wild-type sequence),

with no masking and no marginals, from the LSTM, the GPT and the 3-gram. The
composition-only baseline is the 1-gram, for which s is the sum over substituted
positions of log2 p(new residue) - log2 p(old residue). Each score is compared
with the measured DMS_score (higher = fitter) by Spearman's rho.

  ./scripts/download_proteingym.sh   # assay CSVs into data/raw/proteingym/
  python -m nexttok.variants         # writes results/variants_protein.json
"""

from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from . import ngram
from .data import ROOT, load_protein
from .seqscore import load_checkpoint, neural_logp, ngram_logp, to_ids, to_str

RAW = ROOT / "data" / "raw" / "proteingym"
RES = ROOT / "results"
ASSAYS = (
    "BLAT_ECOLX_Jacquier_2013",                 # organismal fitness, TEM-1 beta-lactamase
    "P53_HUMAN_Kotler_2018",                    # organismal fitness, p53
    "CCDB_ECOLI_Adkar_2012",                    # activity, CcdB toxin
    "NUD15_HUMAN_Suiter_2020",                  # expression (abundance), NUDT15
    "RASK_HUMAN_Weng_2022_binding-DARPin_K55",  # binding, KRAS
    "NUSA_ECOLI_Tsuboyama_2023_1WCL",           # stability, NusA domain (cDNA display proteolysis)
)
# published ProteinGym zero-shot Spearman, for scale (same assays, all mutants): an MSA profile
# model, the two smallest single-sequence models (CARP 600K, UniRep), a small and a large pLM
REFERENCE_MODELS = ("Site-Independent", "CARP (600K)", "Unirep", "ESM2 (8M)", "Progen2 S", "ESM2 (650M)")


# ------------------------------------------------------------------ statistics
def rankdata(x: np.ndarray) -> np.ndarray:
    """Ranks with ties given their average rank (as scipy.stats.rankdata)."""
    x = np.asarray(x, np.float64)
    order = np.argsort(x, kind="mergesort")
    _, first, counts = np.unique(x[order], return_index=True, return_counts=True)
    ranks = np.empty(len(x))
    ranks[order] = np.repeat(first + (counts - 1) / 2.0, counts)
    return ranks


def spearman(a, b) -> float:
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def bootstrap_rho(scores: dict[str, np.ndarray], y: np.ndarray, base: str, B: int = 1000, seed: int = 0) -> dict:
    """95% intervals for rho and for rho - rho(base), resampling mutants (paired)."""
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in scores}
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        for k, s in scores.items():
            draws[k].append(spearman(s[i], y[i]))
    out = {}
    for k, v in draws.items():
        v = np.array(v)
        d = v - np.array(draws[base])
        out[k] = {"ci95": [float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975))],
                  "minus_base_ci95": [float(np.nanquantile(d, 0.025)), float(np.nanquantile(d, 0.975))]}
    return out


# ------------------------------------------------------------------ data
def parse_mutant(m: str) -> list[tuple[str, int, str]]:
    """'A23V:G45D' -> [('A', 22, 'V'), ('G', 44, 'D')] with 0-based positions."""
    out = []
    for part in m.split(":"):
        out.append((part[0], int(part[1:-1]) - 1, part[-1]))
    return out


def reference_rows() -> dict[str, dict]:
    with open(RAW / "DMS_substitutions.csv", newline="") as fh:
        return {r["DMS_id"]: r for r in csv.DictReader(fh)}


def load_assay(dms_id: str, wt: str):
    """Mutants, their sequences and DMS scores, checked against the wild type."""
    muts, seqs, y = [], [], []
    with open(RAW / f"{dms_id}.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            subs = parse_mutant(r["mutant"])
            s = list(wt)
            for a, p, b in subs:
                if wt[p] != a:
                    raise ValueError(f"{dms_id}: {r['mutant']} does not match the wild type")
                s[p] = b
            if "".join(s) != r["mutated_sequence"]:
                raise ValueError(f"{dms_id}: {r['mutant']} disagrees with mutated_sequence")
            muts.append(subs)
            seqs.append(r["mutated_sequence"])
            y.append(float(r["DMS_score"]))
    return muts, seqs, np.array(y)


def composition_scores(muts, unigram: np.ndarray) -> np.ndarray:
    """1-gram log-likelihood ratio: only the substituted residues change."""
    from .vocab import AMINO
    lu = np.log2(unigram)
    return np.array([sum(lu[AMINO.index(b)] - lu[AMINO.index(a)] for a, _, b in m) for m in muts])


def split_membership(wt: str, train_bits: np.ndarray) -> dict:
    """Is the exact wild type in one of our splits, and how many of its 10-mers occur in training?

    `train_bits` is `evaluate.train_kmers()`; its false-positive rate is about 1%.
    """
    from .evaluate import kmer_codes, shared_rate

    where = None
    for split in ("train", "val", "test_seen", "test_novel"):
        d = load_protein(split)
        for o, L in zip(d["offsets"], d["lengths"]):
            if L == len(wt) and to_str(d["tokens"][o + 1: o + 1 + L]) == wt:
                where = split
                break
        if where:
            break
    rate = shared_rate(kmer_codes([to_ids(wt) + 3]), train_bits)
    return {"exact_wild_type_in": where, "kmer10_shared_with_train": rate}


# ------------------------------------------------------------------ main
def run(membership: bool = True) -> dict:
    from .evaluate import protein_train_stream, train_kmers
    from .train import device

    ref = reference_rows()
    train_bits = train_kmers() if membership else None
    pub = {}
    pub_path = RAW / "DMS_substitutions_Spearman_DMS_level.csv"
    if pub_path.exists():
        with open(pub_path, newline="") as fh:
            pub = {r["DMS ID"]: r for r in csv.DictReader(fh)}

    flat, tg = protein_train_stream()
    kn = ngram.KneserNey.fit(flat, tg, 3, 21, 20)
    del flat, tg
    uni, tri = kn.table(1)[0], kn.table(3)
    dev = device()
    models = {name: load_checkpoint(ROOT / "checkpoints" / f"protein_{name}.pt", dev) for name in ("lstm", "gpt")}

    out = {"scoring": "log2 P(mutant) - log2 P(wild type), full sequence, autoregressive",
           "reference_models": list(REFERENCE_MODELS), "assays": {}}
    for dms_id in ASSAYS:
        r = ref[dms_id]
        wt = r["target_seq"]
        assert len(wt) <= 512, dms_id
        muts, seqs, y = load_assay(dms_id, wt)
        ids = [to_ids(s) for s in seqs]
        wt_ids = to_ids(wt)
        scores = {"1-gram": composition_scores(muts, uni)}
        wt_bits = {"1-gram": float(-np.log2(uni[wt_ids]).mean())}
        lp_wt = ngram_logp(tri, 3, [wt_ids])[0]
        scores["3-gram"] = np.array([x.sum() for x in ngram_logp(tri, 3, ids)]) - lp_wt.sum()
        wt_bits["3-gram"] = float(-lp_wt.mean())
        for name, m in models.items():
            lp_wt = neural_logp(m, [wt_ids], dev)[0]
            scores[name] = np.array([x.sum() for x in neural_logp(m, ids, dev, batch=64)]) - lp_wt.sum()
            wt_bits[name] = float(-lp_wt.mean())
        single = np.array([len(m) == 1 for m in muts])
        res = {"selection_type": r["coarse_selection_type"], "length": len(wt), "mutants": int(len(y)),
               "single_mutants": int(single.sum()), "wild_type_bits_per_residue": wt_bits, "spearman": {},
               "spearman_singles": {}}
        boot = bootstrap_rho(scores, y, "1-gram")
        for k, s in scores.items():
            res["spearman"][k] = {"rho": spearman(s, y), **boot[k]}
            res["spearman_singles"][k] = spearman(s[single], y[single])
        if dms_id in pub:
            res["published_spearman"] = {m: float(pub[dms_id][m]) for m in REFERENCE_MODELS if pub[dms_id].get(m)}
        if membership:
            res["split"] = split_membership(wt, train_bits)
        out["assays"][dms_id] = res
        print(dms_id, {k: round(v["rho"], 3) for k, v in res["spearman"].items()}, flush=True)
    names = list(next(iter(out["assays"].values()))["spearman"])
    out["mean_spearman"] = {k: float(np.mean([a["spearman"][k]["rho"] for a in out["assays"].values()]))
                            for k in names}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-membership", action="store_true", help="skip the train-overlap check (needs 0.5 GB)")
    args = ap.parse_args()
    res = run(membership=not args.no_membership)
    (RES / "variants_protein.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res["mean_spearman"], indent=2))


if __name__ == "__main__":
    main()
