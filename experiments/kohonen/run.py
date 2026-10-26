"""Barrido y agregación de resultados, siguiendo compare_activations.py del TP3."""

import argparse
from itertools import product
import multiprocessing
from multiprocessing.pool import ApplyResult, ThreadPool
from pathlib import Path
import time
from typing import List, Tuple, TypedDict

import numpy as np
import pandas as pd
from rich.live import Live
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TaskID, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from experiments.utils import ROOT, KohonenConfig, load_config, prepare_creators, save_json
from experiments.kohonen.plot import plot_results
from models.standarization import ZScoreStandardize
from models.unsupervised import Kohonen, KohonenMetrics

class KohonenResult(KohonenMetrics, TypedDict):
    config_id: int
    grid_size: int
    radius: float
    learning_rate: float
    epochs: int
    seed: int
    seconds: float

def run_task(data: ZScoreStandardize, grid: int, radius: float, eta: float, epochs: int, seed: int, label: int, task: TaskID, progress: Progress) -> Tuple[Kohonen, KohonenResult]:
    start = time.perf_counter()
    kohonen = Kohonen(data, grid, radius, eta, epochs, seed, task, progress)
    return kohonen, {"config_id": label, "grid_size": grid, "radius": radius,
                         "learning_rate": eta, "epochs": epochs, "seed": seed,
                         **kohonen.metrics, "seconds": time.perf_counter() - start}


def validate_config(config: KohonenConfig):
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


def run_study(config_path: Path):
    config: KohonenConfig = load_config(config_path) # type: ignore
    validate_config(config)
    df, data, preprocessing = prepare_creators(
        Path(config["dataset"]), config['id_columns'], config["features"], config["exclude_columns"]
    )
    out = Path(config["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "preprocessing.json", preprocessing)
    save_json(out / "config_used.json", config)
    save_json(out / "metadata.json", {
        "rows": len(df), "features": preprocessing["features"],
        "learning_rate": "constant (original model)",
        "radius": "constant"
    })
    # Exportaciones de exploración: no reemplazan la interpretación del EDA.
    numeric = df[preprocessing["features"]]
    numeric.describe().to_csv(out / "descriptive_statistics.csv")
    numeric.corr().to_csv(out / "correlations.csv")
    standardized = df[["creator_id", "name"]].copy()
    standardized[preprocessing["features"]] = data.standard_values
    standardized.to_csv(out / "standardized_creators.csv", index=False)

    taskprogress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=True,
    )
    globalprogress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    table = Table(box=None)
    table.add_row(taskprogress)
    table.add_row(globalprogress)

    rows: List[KohonenResult] = []
    settings = list(product(config["grid_sizes"], config["radii"], config["learning_rates"], config["epochs"]))
    with (
        ThreadPool(processes=config.get("max_tasks") or int(multiprocessing.cpu_count())) as executor,
        Live(table, refresh_per_second=10),
    ):
        jobs: List[ApplyResult] = []
        for config_id, (grid, radius, rate, epochs) in enumerate(settings):
            for seed in config["seeds"]:
                task = taskprogress.add_task(
                    f"k={grid} r={radius} eta={rate} {seed}",
                    start=False,
                    total=epochs,
                    visible=False,
                    is_task=True,
                )
                jobs.append(
                    executor.apply_async(
                        run_task, (data, grid, radius, rate, epochs, seed, config_id, task, taskprogress)
                    )
                )
        full_progress = globalprogress.add_task(
            f"Total progress ({len(jobs)} elements)", total=len(jobs), is_task=False
        )
        while len(jobs) > 0:
            for job in list(jobs):
                if job.ready():
                    jobs.remove(job)
                    row: Tuple[Kohonen, KohonenResult] = job.get()
                    if row is None:
                        raise ValueError("OOPS")
                    globalprogress.advance(full_progress)
                    rows.append(row[1])
                    # print(f"Configuración {row['config_id']}/{len(settings)} terminada", flush=True)
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
        candidates: pd.DataFrame = trials[trials.config_id == selected.config_id].copy()
        # Mostrar una semilla representativa, cercana a la media, no la mejor corrida.
        representative = candidates.iloc[(candidates.quantization_error - selected.quantization_error_mean).abs().argmin()]
        grid, epochs, seed = (int(representative[k]) for k in ("grid_size", "epochs", "seed"))
        task = globalprogress.add_task(
            f"Candidate k={grid} r={float(representative.radius)} eta={float(representative.learning_rate)}",
            start=False,
            total=epochs,
            visible=False,
            is_task=True,
        )
        job = executor.apply_async(
            run_task, (data, grid, float(representative.radius), float(representative.learning_rate), epochs, seed, 0, task, globalprogress)
        )
        while not job.ready():
            time.sleep(0.1)
        model, _ = job.get()
        bmus = model.bmus
        assignments = df[["creator_id", "name"]].copy()
        assignments["neuron"] = bmus
        assignments["x"] = model.grid_position[bmus, 1]
        assignments["y"] = model.grid_position[bmus, 0]
        assignments.to_csv(out / "assignments.csv", index=False)
        profiles = df[preprocessing["features"]].copy()
        profiles["neuron"] = bmus
        profiles.groupby("neuron").mean().to_csv(out / "neuron_profiles.csv")
        np.savez(out / "selected_model.npz", weights=model.input_weights, grid_size=grid,
                mean=preprocessing["mean"], scale=preprocessing["scale"], features=preprocessing["features"])
        pd.DataFrame(model.u_matrix()).to_csv(out / "u_matrix.csv", index=False)
        print({k: float(representative[k]) for k in representative.index})
        save_json(out / "selection.json", {k: float(representative[k]) for k in representative.index})
    return out


def main():
    parser = argparse.ArgumentParser(description="Comparar Kohonen sobre los creadores del TP4")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/kohonen.json")
    args = parser.parse_args()
    out = run_study(args.config) # type: ignore
    plot_results(out)
    print(f"Resultados y gráficos: {out}")


if __name__ == "__main__":
    main()
