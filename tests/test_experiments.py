import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.metrics import evaluate_som, u_matrix
from experiments.kohonen.run import run_study, train_model
from experiments.utils import ROOT, prepare_creators, transform_features
from models.unsupervised import Kohonen


class ExperimentTests(unittest.TestCase):
    def test_known_metrics_and_u_matrix(self):
        model = Kohonen(np.array([[0.0]]), 2, 1.0, 0.1, 0)
        # Los dos pesos más cercanos al segundo dato son diagonales.
        model.input_weights[:] = [[0], [100], [200], [1]]
        metrics, bmus = evaluate_som(model, np.array([[0.0], [0.5]]))
        self.assertEqual(metrics["quantization_error"], 0.25)
        self.assertEqual(metrics["topographic_error"], 1.0)
        self.assertEqual(metrics["empty_neurons"], 3)
        np.testing.assert_array_equal(bmus, [0, 0])
        np.testing.assert_allclose(u_matrix(model), [[150, 99.5], [199.5, 149]])

    def test_pipeline_seed_reproduces_training_and_restores_global_rng(self):
        data = np.arange(30, dtype=float).reshape(10, 3)
        np.random.seed(999)
        state = np.random.get_state()
        first = train_model(data, 2, 1, 0.3, 10, seed=7)
        restored = np.random.get_state()
        self.assertEqual(state[0], restored[0])
        np.testing.assert_array_equal(state[1], restored[1])
        self.assertEqual(state[2:], restored[2:])
        np.random.seed(123)
        second = train_model(data, 2, 1, 0.3, 10, seed=7)
        np.testing.assert_array_equal(first.input_weights, second.input_weights)

    def test_preprocessing_excludes_ids_and_handles_constants(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data.csv"
            df = pd.DataFrame({"creator_id": ["a", "b"], "name": ["A", "B"],
                               "reach": [10, 30], "constant": [5, 5]})
            df.to_csv(path, index=False)
            _, values, metadata = prepare_creators(path)
            np.testing.assert_array_equal(values, [[-1, 0], [1, 0]])
            self.assertEqual(metadata["constant_features"], ["constant"])
            new = pd.DataFrame({"reach": [50], "constant": [5]})
            np.testing.assert_array_equal(transform_features(new, metadata), [[3, 0]])

    def test_missing_values_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data.csv"
            pd.DataFrame({"creator_id": ["a"], "name": ["A"], "x": [np.nan]}).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                prepare_creators(path)

    def test_full_study_exports_and_reproduces_results(self):
        with tempfile.TemporaryDirectory() as temp:
            config = {"dataset": str(ROOT / "datos/creadores.csv"), "out_dir": temp,
                      "seeds": [1, 2], "grid_sizes": [2], "radii": [1],
                      "learning_rates": [0.1, 0.5], "epochs": [3]}
            out = run_study(config)
            first = pd.read_csv(out / "all_trials.csv").drop(columns="seconds")
            with np.load(out / "selected_model.npz") as saved:
                weights = saved["weights"].copy()
            run_study(config)
            pd.testing.assert_frame_equal(first, pd.read_csv(out / "all_trials.csv").drop(columns="seconds"))
            with np.load(out / "selected_model.npz") as saved:
                np.testing.assert_array_equal(weights, saved["weights"])
            assignments = pd.read_csv(out / "assignments.csv")
            self.assertEqual(len(assignments), 50)
            self.assertEqual(assignments.creator_id.nunique(), 50)
            self.assertEqual(len(first), 4)
            self.assertTrue((out / "preprocessing.json").exists())
            self.assertTrue((out / "metadata.json").exists())


if __name__ == "__main__":
    unittest.main()
