"""Checks for the frame-aware Markov baseline; no downloaded data needed."""

import unittest

import numpy as np
import torch

from nexttok.evaluate import DNA_CTX
from nexttok.frame import (N_CLASS, RC_CLASS, FrameMarkov, by_class, reverse_complement, scoring_windows,
                           train_stream, window_classes)
from nexttok.genome import label_genome
from nexttok.models import Config, build
from nexttok.probe import hidden_states
from nexttok.vocab import BASES, N_SPECIAL

COMP = str.maketrans("ACGT", "TGCA")


def gff_line(start: int, end: int, strand: str, extra: str = "") -> str:
    """One RefSeq-style CDS row; start/end are 0-based half open."""
    return f"chr\tRefSeq\tCDS\t{start + 1}\t{end}\t.\t{strand}\t0\tID=cds-{start};Parent=gene-{start}{extra}"


def planted_genome(rng, n_genes: int = 60, codons: int = 40, spacer: int = 30):
    """Alternating + and - strand genes whose codon position 3 is always G."""
    seq, gff, pos = [], [], 0
    for g in range(n_genes):
        nc = "".join(rng.choice(list(BASES), spacer))
        gene = "".join("".join(rng.choice(list(BASES), 2)) + "G" for _ in range(codons))
        if g % 2:
            gene = gene.translate(COMP)[::-1]  # the same gene on the - strand
        seq += [nc, gene]
        gff.append(gff_line(pos + spacer, pos + spacer + len(gene), "-" if g % 2 else "+"))
        pos += spacer + len(gene)
    return "".join(seq), "\n".join(gff)


def as_dataset(seq: str, gff: str) -> dict:
    enc, labels, _ = label_genome(seq, gff)
    return {"tokens": enc, "labels": labels, "split": np.zeros(len(enc), np.int8)}


class LabelTest(unittest.TestCase):
    """Codon-position bookkeeping on both strands, as used by prepare_dna."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.seq = "".join(rng.choice(list(BASES), 300))
        self.gff = "\n".join([
            gff_line(20, 80, "+"),
            gff_line(150, 210, "-"),
            gff_line(200, 230, "+"),                 # overlaps the - gene by 10 bases
            gff_line(240, 250, "+"),                 # length not a multiple of 3: skipped
            gff_line(260, 290, "+", ";pseudo=true"),  # pseudogene: skipped
        ])
        self.enc, self.labels, self.stats = label_genome(self.seq, self.gff)

    def test_plus_strand_counts_from_the_start(self):
        self.assertEqual(self.labels[20:26].tolist(), [1, 2, 3, 1, 2, 3])
        self.assertEqual(self.labels[77:80].tolist(), [1, 2, 3])
        self.assertEqual(self.labels[19], 0)
        self.assertEqual(self.labels[80], 0)

    def test_minus_strand_counts_from_the_end(self):
        # reading the + strand forwards through a - gene visits codon positions 3, 2, 1
        self.assertEqual(self.labels[150:156].tolist(), [6, 5, 4, 6, 5, 4])
        self.assertEqual(self.labels[195:198].tolist(), [6, 5, 4])  # last whole codon before the overlap

    def test_overlap_and_skips(self):
        self.assertTrue((self.labels[200:210] == 7).all())
        self.assertEqual(self.labels[210:213].tolist(), [2, 3, 1])  # + gene from 200 continues in phase
        self.assertTrue((self.labels[240:300] == 0).all())
        self.assertEqual(self.stats, {"cds_used": 3, "cds_skipped": 2})

    def test_reverse_complement_matches_relabelling_the_other_strand(self):
        n = len(self.seq)
        rc_seq = self.seq.translate(COMP)[::-1]
        rc_gff = "\n".join(gff_line(n - e, n - s, "-" if st == "+" else "+")
                           for s, e, st in ((20, 80, "+"), (150, 210, "-"), (200, 230, "+")))
        enc2, lab2, _ = label_genome(rc_seq, rc_gff)
        sym, lab = reverse_complement(self.enc.astype(np.int64) - N_SPECIAL, self.labels)
        np.testing.assert_array_equal(sym, enc2.astype(np.int64) - N_SPECIAL)
        np.testing.assert_array_equal(lab, lab2)
        np.testing.assert_array_equal(RC_CLASS[RC_CLASS], np.arange(N_CLASS))

    def test_window_hypotheses_match_the_annotation(self):
        k = 12
        for i in list(range(20 + k, 80)) + list(range(150 + k, 200)) + list(range(80 + k, 150)):
            np.testing.assert_array_equal(self.labels[i - k: i], window_classes(k)[self.labels[i]])


class FrameMarkovTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(1)
        cls.seq, cls.gff = planted_genome(rng)
        cls.d = as_dataset(cls.seq, cls.gff)
        flat, tg, classes = train_stream(cls.d, 7)
        cls.flat, cls.tg, cls.classes = flat, tg, classes
        cls.fm = FrameMarkov.fit(flat, tg, classes, max_order=5)
        cls.sym = cls.d["tokens"].astype(np.int64) - N_SPECIAL
        cls.lab = cls.d["labels"].astype(np.int64)

    def test_train_stream_has_both_strands(self):
        n = len(self.seq)
        self.assertEqual(len(self.flat), 2 * n)
        np.testing.assert_array_equal(self.flat[n:], 3 - self.sym[::-1])
        np.testing.assert_array_equal(self.tg, np.concatenate([np.arange(7, n), n + np.arange(7, n)]))

    def test_tables_normalise(self):
        for m, t in self.fm.tables.items():
            self.assertEqual(t.shape, (N_CLASS, 4 ** m, 4))
            self.assertTrue((t > 0).all())
            np.testing.assert_allclose(t.sum(-1), 1.0, atol=1e-9)
        np.testing.assert_allclose(self.fm.prior.sum(), 1.0)
        np.testing.assert_allclose(self.fm.trans.sum(1), 1.0)

    def test_transitions_follow_the_phase_cycle(self):
        A = self.fm.trans
        for a, b in ((1, 2), (2, 3), (6, 5), (5, 4)):  # + runs 1,2,3; - runs 3,2,1 reading forwards
            self.assertGreater(A[a, b], 0.99)
        self.assertGreater(A[3, 1], 0.9)
        self.assertGreater(A[4, 6], 0.9)

    def test_oracle_knows_the_planted_codon_position(self):
        pos = np.arange(10, len(self.sym))
        logp, _ = self.fm.oracle(self.sym, pos, self.lab[pos], 5)
        bits = by_class(logp, self.lab[pos])
        self.assertLess(bits["+ codon 3"], 0.1)   # always G
        self.assertLess(bits["− codon 3"], 0.1)   # always C, read first in each - strand codon
        self.assertGreater(bits["non-coding"], 1.5)  # iid spacers, scored in-sample

    def test_window_posterior_finds_the_frame(self):
        k = 48
        lab = self.lab
        inside = [i for i in range(k, len(lab)) if 0 < lab[i] < 7 and np.all((lab[i - k: i] > 0) & (lab[i - k: i] < 7))]
        inside = np.array(inside)
        W = self.sym[inside[:, None] + np.arange(-k, 1)]
        logp, _, post = self.fm.window(W)
        np.testing.assert_allclose(post.sum(1), 1.0)
        self.assertGreater(post[np.arange(len(inside)), lab[inside]].mean(), 0.9)
        # with the frame found, the planted codon-3 G is predicted almost surely
        self.assertLess(-logp[lab[inside] == 3].mean(), 0.2)

    def test_empty_window_is_the_prior(self):
        _, _, post = self.fm.window(self.sym[:5, None])
        prior = self.fm.prior[:7] / self.fm.prior[:7].sum()
        np.testing.assert_allclose(post, np.tile(prior, (5, 1)))

    def test_hmm_filter_tracks_genes_and_predicts_codon_3(self):
        W = self.sym[None, :]
        logp, _, state = self.fm.hmm(W, 0)
        np.testing.assert_allclose(state.sum(-1), 1.0)
        plus3 = np.flatnonzero(self.lab == 3)
        plus3 = plus3[plus3 > 200]
        self.assertLess(-logp[0, plus3].mean(), 0.2)
        guess = state[0].argmax(1)
        self.assertGreater((guess == self.lab)[self.lab > 0].mean(), 0.8)


class WindowTest(unittest.TestCase):
    def test_scoring_windows_cover_every_scored_target_once(self):
        sym = np.arange(3000) % 4
        segs = [(0, 1200), (2000, 2600)]
        pos = np.concatenate([p for _, p in scoring_windows(sym, segs, T=512)])
        expect = np.concatenate([np.arange(a + DNA_CTX, b) for a, b in segs])
        np.testing.assert_array_equal(np.sort(pos), expect)
        for W, p in scoring_windows(sym, segs, T=512):
            np.testing.assert_array_equal(W[:, DNA_CTX:].reshape(-1), sym[p])


class ProbeFeatureTest(unittest.TestCase):
    def test_hidden_states_end_where_the_model_does(self):
        torch.manual_seed(0)
        x = torch.randint(3, 7, (2, 20))
        lstm = build("lstm", Config(vocab=7, max_len=32)).eval()
        with torch.no_grad():
            top, _ = lstm.rnn(lstm.tok(x))
            self.assertTrue(torch.allclose(hidden_states(lstm, x)[-1], top, atol=1e-5))
            gpt = build("gpt", Config(vocab=7, max_len=32)).eval()
            last = hidden_states(gpt, x)[-1]
            self.assertTrue(torch.allclose(gpt.head(gpt.ln(last)), gpt(x), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
