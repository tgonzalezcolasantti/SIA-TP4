import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.hopfield.data import add_noise, erase_corner, letter_patterns, load_qr_patterns, normalized_overlap
from experiments.hopfield.run import load_config, run_experiments
from experiments.utils import ROOT
from models.hopfield import Hopfield, classify_recall


class HopfieldTests(unittest.TestCase):
    def test_hebb_symmetry_and_diagonal(self):
        patterns = np.array([[1, -1, 1], [-1, -1, 1]])
        model = Hopfield().train(patterns)
        expected = patterns.astype(float).T @ patterns / 3
        np.fill_diagonal(expected, 0)
        np.testing.assert_array_equal(model.weights, expected)
        np.testing.assert_array_equal(model.weights, model.weights.T)
        np.testing.assert_array_equal(np.diag(model.weights), 0)

    def test_single_memory_recovery_and_energy_at_every_update(self):
        original = np.array([1, -1, 1, -1, 1, 1, -1, -1, 1])
        query = original.copy()
        query[:2] *= -1
        model = Hopfield().train([original])
        result = model.recall(query, seed=7, record_history=True, record_updates=True)
        self.assertTrue(result.converged)
        np.testing.assert_array_equal(result.state, original)
        self.assertEqual(classify_recall(result, model.patterns, 0), "correct")
        self.assertTrue(np.all(np.diff(result.update_energies) <= 1e-10))
        self.assertEqual(len(result.update_energies), 1 + len(original) * result.sweeps)
        np.testing.assert_allclose(result.energies, [model.energy(s) for s in result.history], atol=1e-10)

    def test_zero_field_preserves_state(self):
        model = Hopfield().train([[1, 1], [1, -1]])
        result = model.recall([-1, 1], seed=1)
        np.testing.assert_array_equal(result.state, [-1, 1])
        self.assertTrue(result.converged)
        self.assertEqual(result.sweeps, 1)
        self.assertEqual(classify_recall(result, model.patterns), "inverse_memory")

    def test_limit_is_not_reported_as_spurious_or_success(self):
        model = Hopfield().train([[1] * 9])
        query = np.ones(9, dtype=int)
        query[:2] = -1
        result = model.recall(query, max_sweeps=1, seed=0)
        self.assertFalse(result.converged)
        self.assertEqual(classify_recall(result, model.patterns, 0), "not_converged")

    def test_stable_spurious_and_other_memory_are_distinguished(self):
        patterns = np.array([[1, 1, 1, 1], [1, -1, 1, -1],
                             [1, 1, -1, -1], [1, -1, -1, 1]])
        model = Hopfield().train(patterns)
        result = model.recall([1, 1, 1, -1], seed=0)
        self.assertTrue(result.converged)
        self.assertEqual(classify_recall(result, patterns, 0), "spurious")
        self.assertTrue(np.all(result.state * (model.weights @ result.state) >= -1e-12))
        other = model.recall(patterns[1], seed=0)
        self.assertEqual(classify_recall(other, patterns, 0), "other_memory")

    def test_exact_noise_and_corner(self):
        pattern = np.ones(25, dtype=np.int8)
        noisy = add_noise(pattern, 0.2, np.random.default_rng(1))
        self.assertEqual(np.count_nonzero(noisy != pattern), 5)
        self.assertTrue(np.isin(noisy, [-1, 1]).all())
        np.testing.assert_array_equal(pattern, np.ones(25))
        erased = erase_corner(pattern, 0.4).reshape(5, 5)
        np.testing.assert_array_equal(erased[:2, :2], -np.ones((2, 2)))
        self.assertEqual(np.count_nonzero(erased == -1), 4)

    def test_validation(self):
        with self.assertRaises(RuntimeError):
            Hopfield().recall([1])
        for invalid in ([[0, 1]], [[np.nan, 1]], [], [1, -1]):
            with self.assertRaises(ValueError):
                Hopfield().train(invalid)
        model = Hopfield().train([[1, -1]])
        with self.assertRaises(ValueError):
            model.recall([1, 1, 1])
        with self.assertRaises(ValueError):
            model.recall([1, -1], max_sweeps=0)

    def test_qr_shape_and_overlap_without_int8_overflow(self):
        names, patterns, paths = load_qr_patterns(ROOT / "datos/hopfield")
        self.assertEqual(patterns.shape, (5, 1369))
        self.assertEqual(len(names), 5)
        self.assertEqual(len(paths), 5)
        np.testing.assert_allclose(np.diag(normalized_overlap(patterns)), 1)

    def test_reproducible_small_end_to_end_study(self):
        config = load_config(ROOT / "configs/hopfield.json")
        config.update(seeds=[7], noise_levels=[0.0, 0.2], corner_fractions=[],
                      max_sweeps=15, spurious_attempts=20, capacity_letters=["A", "K", "S", "J"])
        with tempfile.TemporaryDirectory() as temp:
            config["out_dir"] = temp
            out = run_experiments(config, mode="both", make_plots=False)
            first = pd.read_csv(out / "qr/all_trials.csv")
            run_experiments(config, mode="both", make_plots=False)
            pd.testing.assert_frame_equal(first, pd.read_csv(out / "qr/all_trials.csv"))
            self.assertEqual(len(first), 10)
            self.assertTrue((out / "letters/capacity_summary.csv").exists())
            self.assertTrue((out / "letters/spurious_search.json").exists())
            with np.load(out / "qr/queries_and_results.npz") as arrays:
                self.assertEqual(arrays["recovered"].shape, (10, 1369))
            self.assertTrue(np.all(first.energy_final <= first.energy_initial + 1e-9))


if __name__ == "__main__":
    unittest.main()
