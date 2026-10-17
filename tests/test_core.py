"""Fast checks that need no downloaded data."""

import unittest

import numpy as np
import torch

import contextlib
import hashlib
import io
import tempfile
from pathlib import Path

from nexttok import ngram
from nexttok.data import segments
from nexttok.evaluate import merge
from nexttok.models import Config, build
from nexttok.prepare import family_bucket, family_key, pack
from nexttok.report import cluster_ci, stats
from nexttok.train import fit, run_name
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

    def test_split_seed_zero_is_the_unsalted_first_release_split(self):
        for key in ("globin family", "protein kinase superfamily", "abc transporter superfamily"):
            h = int(hashlib.sha1(key.encode()).hexdigest(), 16) % 100
            self.assertEqual(family_bucket(key, 0), "train" if h < 80 else ("val" if h < 90 else "test_novel"))

    def test_split_seeds_hold_out_different_families(self):
        keys = [f"family {i}" for i in range(3000)]
        held = [{k for k in keys if family_bucket(k, s) == "test_novel"} for s in (0, 1, 2)]
        for h in held:
            self.assertAlmostEqual(len(h) / len(keys), 0.10, delta=0.02)
        for a in range(3):
            for b in range(a + 1, 3):  # independent 10% draws overlap in about 1% of families
                self.assertLess(len(held[a] & held[b]) / len(keys), 0.03)

    def test_run_names(self):
        self.assertEqual(run_name("protein", "gpt", 2, 1), "protein_gpt_split2_seed1")
        self.assertEqual(run_name("dna", "lstm", 0, 2), "dna_lstm_seed2")

    def test_pack_layout(self):
        p = pack([np.array([5, 6], np.uint8), np.array([7], np.uint8)])
        self.assertEqual(p["tokens"].tolist(), [BOS, 5, 6, EOS, BOS, 7, EOS])
        self.assertEqual(p["offsets"].tolist(), [0, 4])

    def test_segments(self):
        self.assertEqual(segments(np.array([0, 1, 1, 0, 1], bool)), [(1, 3), (4, 5)])


class IntervalTest(unittest.TestCase):
    def test_stats_uses_sample_sd(self):
        st = stats([1.0, 2.0, 3.0])
        self.assertAlmostEqual(st["mean"], 2.0)
        self.assertAlmostEqual(st["sd"], 1.0)
        self.assertEqual(stats([4.0])["sd"], 0.0)

    def test_cluster_bootstrap(self):
        cl = np.repeat(np.arange(50), 20)
        ci = cluster_ci(np.full(len(cl), 0.25), cl)
        self.assertAlmostEqual(ci["lo"], 0.25)
        self.assertAlmostEqual(ci["hi"], 0.25)
        rng = np.random.default_rng(0)
        d = np.repeat(rng.normal(0, 1, 50), 20)  # all variation is between clusters
        ci = cluster_ci(d, cl)
        self.assertLess(ci["lo"], ci["est"])
        self.assertGreater(ci["hi"], ci["est"])
        self.assertEqual(ci["clusters"], 50)
        # resampling single targets would treat the 1000 copies as independent
        self.assertGreater(ci["hi"] - ci["lo"], 2 * 1.96 * d.std() / np.sqrt(len(d)))

    def test_merge_keeps_earlier_seeds(self):
        old = {"test_novel": {"gpt/seed0": {"bits": 4.1}}}
        merge(old, {"test_novel": {"gpt/seed1": {"bits": 4.2}}})
        self.assertEqual(sorted(old["test_novel"]), ["gpt/seed0", "gpt/seed1"])


class Crash(Exception):
    pass


class FakeSampler:
    """Random token windows from a seeded Generator, like ProteinSampler; can crash mid-run."""

    def __init__(self, seed, crash_at=None):
        self.rng, self.crash_at, self.n = np.random.default_rng(seed), crash_at, 0

    def batch(self, B):
        if self.n == self.crash_at:
            raise Crash
        self.n += 1
        x = torch.from_numpy(self.rng.integers(3, 9, (B, 13)))
        return x[:, :-1], x[:, 1:]


class ResumeTest(unittest.TestCase):
    N = 4  # crash after N updates, i.e. at the batch for step N

    def run_fit(self, name, tmp, sampler, resume=False):
        torch.manual_seed(0)
        cfg = Config(vocab=9, max_len=12, d_model=16, n_layer=1, n_head=2, lstm_hidden=16)
        model = build(name, cfg)
        val = [FakeSampler(99).batch(4) for _ in range(2)]
        log = {"name": name, "history": []}
        with contextlib.redirect_stdout(io.StringIO()):
            fit(model, sampler, val, steps=2 * self.N, eval_every=2, lr=1e-2, batch=4, dev=torch.device("cpu"),
                best_path=tmp / "best.pt", last_path=tmp / "last.pt", log=log, meta={}, resume=resume)
        return model, log

    def test_resumed_run_matches_straight_run(self):
        for name in ("gpt", "lstm"):
            with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
                straight, log_s = self.run_fit(name, Path(a), FakeSampler(1))
                with self.assertRaises(Crash):
                    self.run_fit(name, Path(b), FakeSampler(1, crash_at=self.N))
                self.assertEqual(torch.load(Path(b) / "last.pt")["step"], self.N)
                resumed, log_r = self.run_fit(name, Path(b), FakeSampler(1), resume=True)
                for (k, p), q in zip(straight.state_dict().items(), resumed.state_dict().values()):
                    self.assertTrue(torch.equal(p, q), f"{name}: {k} differs after resume")
                strip = lambda h: [(e["step"], e["val_bits"], e["tokens"]) for e in h]
                self.assertEqual(strip(log_s["history"]), strip(log_r["history"]))
                self.assertEqual(log_s["best_val_bits"], log_r["best_val_bits"])
                self.assertEqual(log_r["resumed_at"], [self.N])
                best_s, best_r = (torch.load(Path(d) / "best.pt")["state"] for d in (a, b))
                self.assertTrue(all(torch.equal(best_s[k], best_r[k]) for k in best_s))

    def test_resume_refuses_different_settings(self):
        with tempfile.TemporaryDirectory() as a:
            self.run_fit("gpt", Path(a), FakeSampler(1))
            st = torch.load(Path(a) / "last.pt")
            st["run"]["name"] = "other"
            torch.save(st, Path(a) / "last.pt")
            with self.assertRaises(SystemExit):
                self.run_fit("gpt", Path(a), FakeSampler(1), resume=True)


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
