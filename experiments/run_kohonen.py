"""Barrido y agregación de resultados, siguiendo compare_activations.py del TP3."""

import argparse
from contextlib import redirect_stdout
import hashlib
import io
from itertools import product
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

from experiments.metrics import evaluate_som, u_matrix
from experiments.utils import ROOT, load_config, prepare_creators, save_json
from models.unsupervised import Kohonnen


def train_model(data, grid, radius, rate, epochs, seed):
    """Controla la inicialización aleatoria sin modificar el modelo original."""
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        with redirect_stdout(io.StringIO()):
            return Kohonnen(data, grid, radius, rate, epochs)
    finally:
        np.random.set_state(state)


def validate_config(config):
    for key in ("seeds", "grid_sizes", "radii", "learning_rates", "epochs"):
        values = config[key]
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise ValueError(f"{key}: se requiere una lista no vacía sin duplicados")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) for v in values):
            raise ValueError(f"{key}: valores numéricos finitos requeridos")
    for key, minimum in (("grid_sizes", 2), ("epochs", 1), ("seeds", 0)):
        if any(not isinstance(v, int) or v < minimum for v in config[key]):
            raise ValueError(f"{key}: enteros >= {minimum} requeridos")
    if any(seed > 2**32 - 1 for seed in config["seeds"]):
        raise ValueError("seeds debe contener enteros <= 2**32 - 1")
    if any(not 0 < v <= 1 for v in config["learning_rates"]):
        raise ValueError("learning_rates debe estar en (0, 1]")
    if any(not 0 < v <= min(config["grid_sizes"]) for v in config["radii"]):
        raise ValueError("Cada radio debe ser positivo y <= a todos los tamaños de grilla")


def run_study(config):
    validate_config(config)
    df, data, preprocessing = prepare_creators(
        config["dataset"], config.get("features"), config.get("exclude_columns")
    )
    out = Path(config["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "preprocessing.json", preprocessing)
    save_json(out / "config_used.json", config)
    save_json(out / "metadata.json", {
        "dataset_sha256": hashlib.sha256(Path(config["dataset"]).read_bytes()).hexdigest(),
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "rows": len(df), "features": preprocessing["features"],
        "learning_rate_schedule": "constant (original model)",
        "radius_schedule": "constant", "topology_metrics": "4 orthogonal neighbors",
        "training_order": "original CSV order",
        "model_note": "Original Kohonnen excludes the winning neuron from updates",
        "selection": "min mean quantization error, tie-break mean topographic error",
    })
    # Exportaciones de exploración: no reemplazan la interpretación del EDA.
    numeric = df[preprocessing["features"]]
    numeric.describe().to_csv(out / "descriptive_statistics.csv")
    numeric.corr().to_csv(out / "correlations.csv")
    standardized = df[["creator_id", "name"]].copy()
    standardized[preprocessing["features"]] = data
    standardized.to_csv(out / "standardized_creators.csv", index=False)

    rows = []
    settings = list(product(config["grid_sizes"], config["radii"], config["learning_rates"], config["epochs"]))
    for config_id, (grid, radius, rate, epochs) in enumerate(settings):
        for seed in config["seeds"]:
            start = time.perf_counter()
            model = train_model(data, grid, radius, rate, epochs, seed)
            metrics, _ = evaluate_som(model, data)
            rows.append({"config_id": config_id, "grid_size": grid, "radius": radius,
                         "learning_rate": rate, "epochs": epochs, "seed": seed,
                         **metrics, "seconds": time.perf_counter() - start})
        print(f"Configuración {config_id + 1}/{len(settings)} terminada", flush=True)
    trials = pd.DataFrame(rows)
    trials.to_csv(out / "all_trials.csv", index=False)
    keys = ["config_id", "grid_size", "radius", "learning_rate", "epochs"]
    metrics = ["quantization_error", "topographic_error", "occupancy", "empty_neurons", "seconds"]
    summary = trials.groupby(keys)[metrics].agg(["mean", "std"]).fillna(0)
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index().sort_values(
        ["quantization_error_mean", "topographic_error_mean", "config_id"], kind="stable"
    )
    summary.to_csv(out / "summary.csv", index=False)
    selected = summary.iloc[0]
    candidates = trials[trials.config_id == selected.config_id].copy()
    # Mostrar una semilla representativa, cercana a la media, no la mejor corrida.
    representative = candidates.iloc[(candidates.quantization_error - selected.quantization_error_mean).abs().argmin()]
    grid, epochs, seed = (int(representative[k]) for k in ("grid_size", "epochs", "seed"))
    model = train_model(data, grid, float(representative.radius), float(representative.learning_rate), epochs, seed)
    _, bmus = evaluate_som(model, data)
    assignments = df[["creator_id", "name"]].copy()
    assignments["neuron"] = bmus
    assignments["x"] = bmus % grid
    assignments["y"] = bmus // grid
    assignments.to_csv(out / "assignments.csv", index=False)
    profiles = df[preprocessing["features"]].copy()
    profiles["neuron"] = bmus
    profiles.groupby("neuron").mean().to_csv(out / "neuron_profiles.csv")
    np.savez(out / "selected_model.npz", weights=model.input_weights, grid_size=grid,
             mean=preprocessing["mean"], scale=preprocessing["scale"], features=preprocessing["features"])
    pd.DataFrame(u_matrix(model)).to_csv(out / "u_matrix.csv", index=False)
    save_json(out / "selection.json", {k: representative[k].item() for k in representative.index})
    return out


def main():
    parser = argparse.ArgumentParser(description="Comparar Kohonen sobre los creadores del TP4")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/kohonen.json")
    args = parser.parse_args()
    out = run_study(load_config(args.config))
    from experiments.plot_results import plot_results
    plot_results(out)
    print(f"Resultados y gráficos: {out}")


if __name__ == "__main__":
    main()
