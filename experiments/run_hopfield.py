"""Experimentos reproducibles de letras y QR, independientes de Kohonen."""

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from experiments.hopfield_data import (
    add_noise, erase_corner, letter_patterns, load_qr_patterns, normalized_overlap,
)
from experiments.plot_hopfield import (
    capacity_figure, overview, pattern_image, recovery_figure, summary_figures,
)
from experiments.utils import ROOT, save_json
from models.hopfield import Hopfield, classify_recall


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("qr_dir", "out_dir"):
        config[key] = str((path.parent / config[key]).resolve())
    for key in ("seeds", "noise_levels"):
        values = config[key]
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise ValueError(f"{key}: usar una lista no vacía sin duplicados")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in config["seeds"]):
        raise ValueError("Las semillas deben ser enteros no negativos")
    for key in ("noise_levels", "corner_fractions"):
        values = config[key]
        if not isinstance(values, list) or len(set(values)) != len(values):
            raise ValueError(f"{key}: usar una lista sin duplicados")
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not np.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError(f"{key}: valores finitos entre 0 y 1")
    for key in ("max_sweeps", "spurious_attempts"):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key}: entero positivo requerido")
    for key in ("capacity_noise", "spurious_noise"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{key}: valor finito entre 0 y 1")
    letter_patterns(config["letters"])
    letter_patterns(config["capacity_letters"])
    if len(config["letters"]) != 4:
        raise ValueError("El experimento base de letras debe almacenar cuatro patrones")
    return config


def trial_seeds(seed, *indices):
    values = np.random.SeedSequence([seed, *indices]).generate_state(2, dtype=np.uint64)
    return int(values[0]), int(values[1])


def run_patterns(names, patterns, side, out, config, make_plots=True):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    examples = out / "examples"
    examples.mkdir(exist_ok=True)
    model = Hopfield().train(patterns)
    np.savez_compressed(out / "stored_patterns.npz", names=names, patterns=patterns, side=side)
    pd.DataFrame(normalized_overlap(patterns), index=names, columns=names).to_csv(out / "overlap.csv")
    stability = []
    for name, pattern in zip(names, patterns):
        stability.append({"pattern": name, "stable": bool(np.all(pattern * (model.weights @ pattern) >= -1e-12)),
                          "energy": model.energy(pattern)})
    pd.DataFrame(stability).to_csv(out / "stored_stability.csv", index=False)

    rows, traces, queries, recovered = [], [], [], []
    damages = [("random_flips", config["noise_levels"]), ("white_corner", config["corner_fractions"])]
    for damage_idx, (damage, levels) in enumerate(damages):
        for level_idx, level in enumerate(levels):
            for target, (name, original) in enumerate(zip(names, patterns)):
                for seed in config["seeds"]:
                    query_seed, recall_seed = trial_seeds(seed, side, damage_idx, level_idx, target)
                    query = (add_noise(original, level, np.random.default_rng(query_seed))
                             if damage == "random_flips" else erase_corner(original, level))
                    representative = seed == config["seeds"][0] and (
                        (damage == "random_flips" and level_idx == len(levels) // 2) or target == 0
                    )
                    result = model.recall(query, max_sweeps=config["max_sweeps"], seed=recall_seed,
                                          record_history=representative, record_updates=representative)
                    trial_id = len(rows)
                    status = classify_recall(result, patterns, target)
                    hamming = int(np.count_nonzero(result.state != original))
                    rows.append({"trial_id": trial_id, "pattern": name, "damage": damage, "level": level,
                                 "seed": seed, "query_seed": query_seed, "recall_seed": recall_seed,
                                 "hamming_input": int(np.count_nonzero(query != original)),
                                 "hamming_final": hamming, "success": status == "correct",
                                 "converged": result.converged, "status": status, "sweeps": result.sweeps,
                                 "energy_initial": result.energies[0], "energy_final": result.energies[-1]})
                    for sweep, energy in enumerate(result.energies):
                        traces.append({"trial_id": trial_id, "sweep": sweep, "energy": energy,
                                       "changed_neurons": 0 if sweep == 0 else result.changes[sweep - 1]})
                    queries.append(query)
                    recovered.append(result.state)
                    if representative:
                        prefix = examples / f"trial_{trial_id:04d}"
                        np.savez_compressed(str(prefix) + ".npz", original=original, query=query,
                                            recovered=result.state, history=result.history)
                        pd.DataFrame({"neuron_update": range(len(result.update_energies)),
                                      "energy": result.update_energies}).to_csv(str(prefix) + "_updates.csv", index=False)
                        if make_plots:
                            recovery_figure(original, query, result, side, str(prefix) + ".png",
                                            f"{name}: {damage}, nivel={level:g}, {status}")
                            pattern_image(result.state, str(prefix) + "_recovered.png", side)
            print(f"{out.name}: {damage} {level:g} terminado", flush=True)
    trials = pd.DataFrame(rows)
    trials.to_csv(out / "all_trials.csv", index=False)
    pd.DataFrame(traces).to_csv(out / "recall_traces.csv", index=False)
    np.savez_compressed(out / "queries_and_results.npz", queries=queries, recovered=recovered)
    summary = trials.groupby(["damage", "level"], sort=True).agg(
        trials=("success", "size"), success_rate=("success", "mean"),
        convergence_rate=("converged", "mean"), sweeps_mean=("sweeps", "mean"),
        hamming_final_mean=("hamming_final", "mean"),
    ).reset_index()
    summary.to_csv(out / "summary.csv", index=False)
    trials.groupby(["pattern", "damage", "level"]).agg(
        success_rate=("success", "mean"), sweeps_mean=("sweeps", "mean"),
        hamming_final_mean=("hamming_final", "mean"),
    ).to_csv(out / "per_pattern_summary.csv")
    trials.groupby(["damage", "level", "status"]).size().rename("count").to_csv(out / "outcomes.csv")
    if make_plots:
        overview(out, names, patterns, side)
        summary_figures(out)
    return trials


def run_capacity(out, config, make_plots=True):
    names = config["capacity_letters"]
    all_patterns = letter_patterns(names)
    rows = []
    for count in range(1, len(names) + 1):
        patterns = all_patterns[:count]
        model = Hopfield().train(patterns)
        for target, original in enumerate(patterns):
            for seed in config["seeds"]:
                noise_seed, recall_seed = trial_seeds(seed, 100, count, target)
                query = add_noise(original, config["capacity_noise"], np.random.default_rng(noise_seed))
                result = model.recall(query, max_sweeps=config["max_sweeps"], seed=recall_seed)
                rows.append({"stored_count": count, "pattern": names[target], "seed": seed,
                             "success": classify_recall(result, patterns, target) == "correct",
                             "converged": result.converged, "sweeps": result.sweeps})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "capacity_trials.csv", index=False)
    frame.groupby("stored_count").agg(trials=("success", "size"), success_rate=("success", "mean"),
                                      convergence_rate=("converged", "mean")).to_csv(out / "capacity_summary.csv")
    if make_plots:
        capacity_figure(out)


def find_spurious(out, config, make_plots=True):
    # Evitar que un ejemplo de una ejecución anterior parezca un hallazgo nuevo.
    for filename in ("spurious.npz", "spurious_energy.csv", "spurious.png"):
        (out / filename).unlink(missing_ok=True)
    patterns = letter_patterns(config["letters"])
    model = Hopfield().train(patterns)
    attempts = []
    found = None
    for attempt in range(config["spurious_attempts"]):
        target = attempt % len(patterns)
        noise_seed, recall_seed = trial_seeds(config["seeds"][0], 200, attempt)
        query = add_noise(patterns[target], config["spurious_noise"], np.random.default_rng(noise_seed))
        result = model.recall(query, max_sweeps=config["max_sweeps"], seed=recall_seed,
                              record_history=True, record_updates=True)
        status = classify_recall(result, patterns, target)
        attempts.append({"attempt": attempt, "target": config["letters"][target],
                         "query_seed": noise_seed, "recall_seed": recall_seed, "status": status})
        if status == "spurious":
            found = {"found": True, "attempt": attempt, "target": config["letters"][target],
                     "hamming_input": int(np.count_nonzero(query != patterns[target])),
                     "sweeps": result.sweeps, "energy": result.energies[-1],
                     "hamming_to_memories": np.count_nonzero(patterns != result.state, axis=1).tolist()}
            np.savez_compressed(out / "spurious.npz", query=query, recovered=result.state, history=result.history)
            pd.DataFrame({"neuron_update": range(len(result.update_energies)),
                          "energy": result.update_energies}).to_csv(out / "spurious_energy.csv", index=False)
            if make_plots:
                recovery_figure(patterns[target], query, result, 5, out / "spurious.png", "Estado espurio estable")
            break
    pd.DataFrame(attempts).to_csv(out / "spurious_search.csv", index=False)
    save_json(out / "spurious_search.json", found or {"found": False, "attempts": len(attempts)})


def run_experiments(config, mode="both", make_plots=True):
    if mode not in {"letters", "qr", "both"}:
        raise ValueError("mode debe ser letters, qr o both")
    out = Path(config["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "config_used.json", config)
    save_json(out / "metadata.json", {
        "mode": mode, "python": platform.python_version(), "numpy": np.__version__,
        "update": "asynchronous, random permutation without replacement each sweep",
        "tie_rule": "preserve state for abs(field) <= 1e-12", "black": 1, "white": -1,
        "noise": "floor(fraction*N) distinct bit inversions", "corner": "white top-left square",
        "energy": "-0.5 * state.T @ weights @ state", "weights": "patterns.T @ patterns / N, zero diagonal",
        "iteration": "one complete sweep of all neurons", "qr_decoding": "not performed",
    })
    if mode in {"letters", "both"}:
        letters_out = out / "letters"
        run_patterns(config["letters"], letter_patterns(config["letters"]), 5, letters_out, config, make_plots)
        run_capacity(letters_out, config, make_plots)
        find_spurious(letters_out, config, make_plots)
    if mode in {"qr", "both"}:
        names, patterns, paths = load_qr_patterns(config["qr_dir"])
        save_json(out / "qr_input_hashes.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
        run_patterns(names, patterns, 37, out / "qr", config, make_plots)
    print(f"Resultados de Hopfield: {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hopfield: letras 5x5 y QR 37x37")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/hopfield.json")
    parser.add_argument("--mode", choices=["letters", "qr", "both"], default="both")
    parser.add_argument("--no-plots", action="store_true", help="Exportar datos y trazas sin generar PNG")
    args = parser.parse_args()
    run_experiments(load_config(args.config), args.mode, not args.no_plots)
