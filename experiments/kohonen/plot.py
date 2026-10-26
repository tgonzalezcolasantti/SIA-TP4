"""Gráficos desde resultados guardados, siguiendo ej2/plot_comparisons.py del TP3."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.utils import ROOT


def plot_results(results_dir):
    out = Path(results_dir)
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    def save_figure(fig, name):
        fig.tight_layout()
        fig.savefig(plots / name, dpi=150, bbox_inches="tight")
        plt.close(fig)

    summary = pd.read_csv(out / "summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels = [f"k={int(r.grid_size)} R={r.radius:g} η={r.learning_rate:g}"
              for r in summary.itertuples()]
    for ax, metric, title in zip(axes, ["quantization_error", "topographic_error"],
                                ["Error de cuantización", "Error topográfico"]):
        ax.bar(np.arange(len(summary)), summary[f"{metric}_mean"], yerr=summary[f"{metric}_std"], capsize=3)
        ax.set_xticks(np.arange(len(summary)), labels, rotation=90)
        ax.set_title(title)
    save_figure(fig, "comparison.png")

    with np.load(out / "selected_model.npz") as model:
        grid = int(model["grid_size"])
        weights, features = model["weights"], model["features"]
    assignments = pd.read_csv(out / "assignments.csv")
    counts = np.bincount(assignments.neuron, minlength=grid**2).reshape(grid, grid)
    u = pd.read_csv(out / "u_matrix.csv").to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(max(12, grid * 3), max(6, grid * 1.5)))
    for ax, matrix, title in zip(axes, [counts, u], ["Creadores por neurona", "Matriz U (vecinos ortogonales)"]):
        im = ax.imshow(matrix, cmap="viridis")
        fig.colorbar(im, ax=ax)
        ax.set_title(title)
        ax.set_xticks(range(grid)); ax.set_yticks(range(grid))
        ax.set_xlabel("x"); ax.set_ylabel("y")
    for (x, y), group in assignments.groupby(["x", "y"]):
        axes[0].text(x, y, "\n".join(group.creator_id), ha="center", va="center", fontsize=7,
                     bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
    save_figure(fig, "map_and_u_matrix.png")

    columns = min(4, len(features))
    rows = int(np.ceil(len(features) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 4, rows * 3.5), squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i >= len(features):
            ax.axis("off")
            continue
        im = ax.imshow(weights[:, i].reshape(grid, grid), cmap="coolwarm")
        ax.set_title(features[i], fontsize=9)
        fig.colorbar(im, ax=ax, label="Peso estandarizado")
    save_figure(fig, "component_planes.png")

    correlations = pd.read_csv(out / "correlations.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(correlations, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(correlations)), correlations.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(correlations)), correlations.index, fontsize=7)
    ax.set_title("Correlaciones de las variables seleccionadas")
    fig.colorbar(im, ax=ax)
    save_figure(fig, "correlations.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerar gráficos sin entrenar")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results/kohonen")
    plot_results(parser.parse_args().results_dir)
