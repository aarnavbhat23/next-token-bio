"""Fast checks that need no downloaded data."""

import unittest

import numpy as np
import torch

from nexttok import ngram
from nexttok.data import segments
from nexttok.models import Config, build
from nexttok.prepare import family_bucket, family_key, pack
from nexttok.vocab import BOS, EOS, N_SPECIAL, decode, encode_table


class KneserNeyTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.flat = rng.integers(0, 4, 30000)
        self.flat[::5] = 2  # some real structure
        self.model = ngram.KneserNey.fit(self.flat, np.arange(6, len(self.flat)), 6, 4, 4)

    def test_every_table_is_a_distribution(self):
        for o in range(1, 7):
            t = self.model.table(o)
            self.assertEqual(t.shape, (4 ** (o - 1), 4))
            self.assertTrue((t > 0).all())
            np.testing.assert_allclose(t.sum(1), 1.0, atol=1e-9)

    def test_context_index_uses_oldest_symbol_as_most_significant(self):
        flat = np.array([1, 2, 3, 0])
        self.assertEqual(ngram.context_index(flat, np.array([3]), 3, 4)[0], 1 * 16 + 2 * 4 + 3)


class SplitTest(unittest.TestCase):
    def test_family_key_groups_at_first_clause(self):
        self.assertEqual(family_key("Protein kinase superfamily, Ser/Thr protein kinase family"),
                         "protein kinase superfamily")

    def test_family_bucket_is_deterministic(self):
        self.assertEqual(family_bucket("globin family"), family_bucket("globin family"))

    def test_pack_layout(self):
        p = pack([np.array([5, 6], np.uint8), np.array([7], np.uint8)])
        self.assertEqual(p["tokens"].tolist(), [BOS, 5, 6, EOS, BOS, 7, EOS])
        self.assertEqual(p["offsets"].tolist(), [0, 4])

    def test_segments(self):
        self.assertEqual(segments(np.array([0, 1, 1, 0, 1], bool)), [(1, 3), (4, 5)])


class VocabTest(unittest.TestCase):
    def test_roundtrip_and_reverse_complement(self):
        enc = encode_table("dna")[np.frombuffer(b"ACGT", np.uint8)]
        self.assertEqual(decode("dna", enc), "ACGT")
        rc = (2 * N_SPECIAL + 3) - enc[::-1]  # same rule as DNASampler
        self.assertEqual(decode("dna", rc), "ACGT")  # ACGT is its own reverse complement
        rc = (2 * N_SPECIAL + 3) - encode_table("dna")[np.frombuffer(b"AACG", np.uint8)][::-1]
        self.assertEqual(decode("dna", rc), "CGTT")


class ModelTest(unittest.TestCase):
    def test_gpt_is_causal(self):
        torch.manual_seed(0)
        m = build("gpt", Config(vocab=23, max_len=16)).eval()
        x = torch.randint(3, 23, (1, 16))
        y = x.clone()
        y[0, 10:] = 3
        with torch.no_grad():
            self.assertTrue(torch.allclose(m(x)[0, :10], m(y)[0, :10], atol=1e-5))

    def test_models_are_parameter_matched(self):
        g = sum(p.numel() for p in build("gpt", Config(vocab=23, max_len=512)).parameters())
        l = sum(p.numel() for p in build("lstm", Config(vocab=23, max_len=512)).parameters())
        self.assertLess(abs(g - l) / g, 0.1)


if __name__ == "__main__":
    unittest.main()
