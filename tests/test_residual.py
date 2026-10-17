"""Checks for the residual, homology and variant-effect code; no downloaded data needed."""

import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from nexttok import homology, ngram, residual, seqscore, variants
from nexttok.models import Config, build
from nexttok.vocab import AMINO


def ids(s: str) -> np.ndarray:
    return np.array([AMINO.index(c) for c in s], np.int64)


class FeatureTest(unittest.TestCase):
    def test_hydrophobic_classes(self):
        for a in AMINO:
            want = 0 if a in residual.HYDROPHOBIC else 1
            self.assertEqual(residual.HYDRO2[AMINO.index(a)], want, a)
        self.assertEqual(residual.HYDRO3[AMINO.index("K")], 2)
        self.assertEqual(residual.HYDRO3[AMINO.index("S")], 1)
        self.assertEqual(residual.HYDRO3[AMINO.index("L")], 0)

    def test_lag_pattern_reads_the_right_residues(self):
        # context ends ...x7 . . x4 x3 x2 x1, target follows; lag L is ctx[-L]
        ctx = ids("KKKLKKLLKS")[None].astype(np.uint8)  # lag1=S lag2=K lag3=L lag4=L lag7=L
        idx, n = residual.lag_pattern(ctx, (2, 3, 4, 7), residual.HYDRO2)
        self.assertEqual(n, 16)
        self.assertEqual(idx[0], int("1000", 2))  # K=1 (not hydrophobic), L=0, L=0, L=0

    def test_window_counts_match_brute_force(self):
        rng = np.random.default_rng(0)
        ctx = rng.integers(0, 20, (50, 64)).astype(np.uint8)
        got = residual.window_counts(ctx, (1, 8, 64))
        for w in (1, 8, 64):
            want = np.stack([np.bincount(r[-w:], minlength=20) for r in ctx])
            np.testing.assert_array_equal(got[w], want)

    def test_hydrophobic_moment_separates_helix_and_strand_periodicity(self):
        j = np.arange(18)
        helix = np.cos(np.radians(100) * j)
        strand = np.cos(np.radians(170) * j)
        self.assertGreater(residual.hydrophobic_moment(helix, 100), 3 * residual.hydrophobic_moment(helix, 170))
        self.assertGreater(residual.hydrophobic_moment(strand, 170), 3 * residual.hydrophobic_moment(strand, 100))
        self.assertAlmostEqual(residual.hydrophobic_moment(np.ones(5), 0), 1.0)

    def test_structure_labels_ignore_the_target_residue(self):
        rng = np.random.default_rng(2)
        seqs = [rng.integers(0, 20, 60) for _ in range(30)]
        sid, pos = np.arange(30), np.full(30, 30)
        a = residual.structure_labels(seqs, sid, pos)
        for s in seqs:
            s[30] = (s[30] + 7) % 20  # change only the residue being predicted
        b = residual.structure_labels(seqs, sid, pos)
        for k in ("mu100", "mu170"):
            np.testing.assert_array_equal(a[k], b[k])
        np.testing.assert_array_equal(a["hydropathy"], b["hydropathy"])


class TargetTest(unittest.TestCase):
    def test_same_draw_as_the_context_sweep(self):
        """Mirror of evaluate.sweep_targets' candidate list and RNG call."""
        rng = np.random.default_rng(3)
        seqs = [rng.integers(0, 20, n) for n in (5, 40, 12, 33, 9, 60)]
        k, n = 10, 30
        cand = [(i, p) for i, s in enumerate(seqs) for p in range(k, len(s))]
        pick = np.random.default_rng(7).choice(len(cand), size=n, replace=False)
        ctx, tgt, sid, pos = residual.long_context_targets(seqs, n, 7, k)
        for r, c in enumerate(pick):
            i, p = cand[c]
            self.assertEqual((sid[r], pos[r]), (i, p))
            np.testing.assert_array_equal(ctx[r], seqs[i][p - k: p])
            self.assertEqual(tgt[r], seqs[i][p])


    def test_packed_targets_read_the_packed_stream(self):
        from nexttok.prepare import pack
        rng = np.random.default_rng(5)
        seqs = [rng.integers(0, 20, n) for n in (5, 40, 12, 33)]
        d = pack([(s + 3).astype(np.uint8) for s in seqs])
        ctx, tgt = residual.packed_targets(d, 100, 0, k=10)
        self.assertEqual(len(tgt), (40 - 10) + (12 - 10) + (33 - 10))  # every candidate, no repeats
        windows = {tuple(s[p - 10: p + 1]) for s in seqs for p in range(10, len(s))}
        for c, t in zip(ctx, tgt):
            self.assertIn(tuple(c.tolist()) + (t,), windows)


class FitTest(unittest.TestCase):
    def test_em_recovers_mixture_weight(self):
        rng = np.random.default_rng(0)
        a, b = np.full(20, 0.05), np.r_[np.full(10, 0.09), np.full(10, 0.01)]
        x = np.where(rng.random(40000) < 0.3, rng.choice(20, 40000, p=a), rng.choice(20, 40000, p=b))
        w = residual.fit_mixture(np.stack([a[x], b[x]], 1))
        self.assertAlmostEqual(w[0], 0.3, delta=0.05)
        self.assertAlmostEqual(w.sum(), 1.0)

    def test_loglinear_learns_a_conditional_distribution(self):
        rng = np.random.default_rng(0)
        f = rng.integers(0, 2, 20000)
        p = np.full((2, 20), 0.5 / 18)
        p[0, 0] = p[1, 1] = 0.45
        p[0, 1] = p[1, 0] = 0.05
        t = np.array([rng.choice(20, p=p[v]) for v in f])
        base = np.log(np.full((len(t), 20), 1 / 20))
        W = residual.fit_loglinear(base, [(f, 2)], t, l2=0.0)
        rows = np.exp(residual.apply_loglinear(base, [(f, 2)], W))
        np.testing.assert_allclose(rows.sum(1), 1.0, atol=1e-9)
        self.assertAlmostEqual(rows[f == 0][0, 0], 0.45, delta=0.03)
        self.assertAlmostEqual(rows[f == 1][0, 0], 0.05, delta=0.02)
        # with nothing to learn the correction stays at zero
        W0 = residual.fit_loglinear(np.log(np.exp(residual.apply_loglinear(base, [(f, 2)], W))), [(f, 2)], t, 0.0)
        self.assertLess(np.abs(W0[0]).max(), 0.05)

    def test_cluster_ci_brackets_the_mean(self):
        rng = np.random.default_rng(1)
        x, c = rng.normal(1.0, 1.0, 2000), rng.integers(0, 100, 2000)
        lo, hi = residual.cluster_mean_ci(x, c)
        self.assertLess(lo, x.mean())
        self.assertGreater(hi, x.mean())


class HomologyTest(unittest.TestCase):
    def test_cluster_membership(self):
        rows = [("train|1", "train|1"), ("train|1", "test_novel|4"), ("test_novel|2", "test_novel|2"),
                ("test_novel|2", "test_novel|7"), ("test_novel|9", "test_novel|9"), ("test_novel|9", "train|3")]
        self.assertEqual(homology.novel_sharing_train(rows), {4, 9})

    def test_search_thresholds(self):
        rows = [("test_novel|1", "train|5", "0.31", "0.60", "0.5", "1e-9", "50"),
                ("test_novel|2", "train|5", "0.29", "0.90", "0.5", "1e-9", "50"),
                ("test_novel|3", "train|6", "0.80", "0.40", "0.5", "1e-9", "50")]
        self.assertEqual(homology.novel_with_hits(rows), {1})


class SeqScoreTest(unittest.TestCase):
    def test_string_roundtrip(self):
        s = "MKVLAWCY"
        self.assertEqual(seqscore.to_str(ids(s) + 3), s)
        np.testing.assert_array_equal(seqscore.to_ids(s), ids(s))

    def test_ngram_logp_matches_padded_stream(self):
        rng = np.random.default_rng(0)
        train = rng.integers(0, 20, 5000)
        table = ngram.KneserNey.fit(train, np.arange(2, len(train)), 3, 21, 20).table(3)
        seqs = [rng.integers(0, 20, 7), rng.integers(0, 20, 3)]
        got = seqscore.ngram_logp(table, 3, seqs)
        for s, lp in zip(seqs, got):
            ctx = np.r_[20, 20, s]
            want = [math.log2(table[ctx[i] * 21 + ctx[i + 1], s[i]]) for i in range(len(s))]
            np.testing.assert_allclose(lp, want, rtol=1e-12)

    def test_neural_logp_is_causal_and_normalised(self):
        torch.manual_seed(0)
        m = build("gpt", Config(vocab=23, max_len=32)).eval()
        s = ids("MKVLAWCYEEKL")
        t = s.copy()
        t[6] = AMINO.index("P")
        a, b = seqscore.neural_logp(m, [s, t], torch.device("cpu"))
        np.testing.assert_allclose(a[:6], b[:6], atol=1e-5)  # a substitution only affects itself and later
        self.assertFalse(np.allclose(a[6:], b[6:]))
        self.assertTrue((a < 0).all())


class VariantTest(unittest.TestCase):
    def test_rankdata_averages_ties(self):
        np.testing.assert_array_equal(variants.rankdata([10, 20, 20, 30]), [0, 1.5, 1.5, 3])

    def test_spearman(self):
        x = np.arange(50.0)
        self.assertAlmostEqual(variants.spearman(x, x ** 3), 1.0)
        self.assertAlmostEqual(variants.spearman(x, -x), -1.0)
        try:
            from scipy.stats import spearmanr
        except ImportError:
            return
        rng = np.random.default_rng(0)
        a, b = rng.integers(0, 5, 200), rng.normal(size=200)
        self.assertAlmostEqual(variants.spearman(a, b), spearmanr(a, b).statistic, places=10)

    def test_parse_and_composition_score(self):
        self.assertEqual(variants.parse_mutant("A23V:G45D"), [("A", 22, "V"), ("G", 44, "D")])
        u = np.linspace(1, 2, 20)
        u /= u.sum()
        s = variants.composition_scores([[("A", 0, "W")]], u)
        self.assertAlmostEqual(s[0], math.log2(u[AMINO.index("W")] / u[AMINO.index("A")]))

    def test_load_assay_checks_the_wild_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "X.csv").write_text("mutant,mutated_sequence,DMS_score,DMS_score_bin\n"
                                             "K2A,MAV,0.5,1\nM1L:V3G,LKG,-1.0,0\n")
            with mock.patch.object(variants, "RAW", Path(tmp)):
                muts, seqs, y = variants.load_assay("X", "MKV")
                self.assertEqual(seqs, ["MAV", "LKG"])
                np.testing.assert_array_equal(y, [0.5, -1.0])
                (Path(tmp) / "Y.csv").write_text("mutant,mutated_sequence,DMS_score,DMS_score_bin\nQ2A,MAV,0.5,1\n")
                with self.assertRaises(ValueError):
                    variants.load_assay("Y", "MKV")


if __name__ == "__main__":
    unittest.main()
