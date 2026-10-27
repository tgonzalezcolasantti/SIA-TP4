"""Gráficos desde resultados guardados, siguiendo ej2/plot_comparisons.py del TP3."""

import argparse
import csv
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional

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

    csv_file = results_dir / 'all_trials.csv'
    
    time_vs_triangles_map = parse_csv(csv_file, ['radius'], 'grid_size', 'quantization_error')
    print(time_vs_triangles_map)
    bar_graph(plots / 'error_vs_grid.png', time_vs_triangles_map, "Error de cuantizacion vs grilla", "Error")

    time_vs_triangles_map = parse_csv(csv_file, ['radius'], 'grid_size', 'empty_neurons')
    print(time_vs_triangles_map)
    bar_graph(plots / 'vacant_vs_grid.png', time_vs_triangles_map, "Neuronas vacias vs grilla", "Error")

    time_vs_triangles_map = parse_csv(csv_file, [], 'learning_rate', 'quantization_error')
    print(time_vs_triangles_map)
    bar_graph(plots / 'error_vs_eta.png', time_vs_triangles_map, "Error de cuantizacion vs eta", "Error")

    time_vs_triangles_map = parse_csv(csv_file, ['grid_size'], 'radius', 'quantization_error')
    print(time_vs_triangles_map)
    bar_graph(plots / 'error_vs_rad.png', time_vs_triangles_map, "Error de cuantizacion vs radio", "Error")

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
        axes[0].text(x, y, "\n".join(group.name), ha="center", va="center", fontsize=7,
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

colors = ["lightcoral", "red", "sandybrown", "darkgoldenrod", "yellowgreen", "lightgreen", "chartreuse", "deepskyblue", "royalblue", "mediumblue", "slateblue", "mediumpurple", "hotpink", "lightpink"]

def bar_graph(output_file, data: Mapping[str, Mapping[str, List[float]]], xlabel: str, ylabel: str, sorter: Optional[Callable[[str], float]] = None, log: bool = False):
    bar_width = 1 / (len(data) + 2)
    fig, ax = plt.subplots(figsize =(10, 8))
    groups = list(data[list(data.keys())[0]].keys())

    for idx, x in enumerate(sorted(data.keys(), key=sorter)):
        bars_x = np.arange(len(data[x])) + bar_width * idx
        y_data = [data[x][group] for group in groups]
        # if isinstance(y_data[0], List):
        means = [np.mean(i) for i in y_data] # type: ignore
        errors = [np.std(i) for i in y_data] # type: ignore
        b = plt.bar(bars_x, means, width=bar_width, color = colors[idx],
                        edgecolor='grey', label=x, yerr=errors)
        plt.bar_label(b, [f'{i:.4f}' for i in means], padding=5, rotation=90)
        # else:
        #     b = plt.bar(bars_x, y_data, color=colors[idx], width = bar_width,
        #             edgecolor ='grey', label=x)
        #     plt.bar_label(b, y_data, padding=5, rotation=90)

    plt.xlabel(xlabel, fontweight ='bold', fontsize = 15)
    plt.ylabel(ylabel, fontweight ='bold', fontsize = 15)
    plt.xticks([r + 0.5-bar_width for r in range(len(data[list(data.keys())[0]]))], groups)
    if log:
        ax.set_yscale('log')
    ax.margins(y=0.2)
    plt.legend()
    #plt.show()
    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

def line_graph(output_file: Path, data: Mapping[str, Mapping[str, List[float]]], xlabel: str, ylabel: str):
    _, ax = plt.subplots(figsize =(20, 8))

    for idx, x in enumerate(data.keys()):
        plot_x = [float(i) for _, i in enumerate(sorted(data[x].keys(), key=float))]
        x_order = [idx for idx, _ in enumerate(sorted(data[x].keys(), key=float))]
        print(plot_x)
        print(x_order)
        plot_y = np.mean(list(data[x][str(plot_x[j])] for j in x_order), axis=1)
        plot_yerr = np.std(list(data[x][str(plot_x[j])] for j in x_order), axis=1)
        print(plot_y)
        plt.errorbar(x = plot_x, y = plot_y, yerr=plot_yerr, label=x)

    plt.xlabel(xlabel, fontweight ='bold', fontsize = 15)
    plt.ylabel(ylabel, fontweight ='bold', fontsize = 15)
    # plt.xticks([r + 1 for r in range(len(data))], sorted(data.keys()))
    ax.set_xscale('log')
    ax.margins(y=0.2)
    plt.legend()
    #plt.show()
    plt.savefig(output_file)


def parse_csv(path: Path, x_main: List[str], x_groupby: str | None, y: str, filter_list: Optional[Dict[str, str]] = None) -> Dict[str, Dict[str, List[float]]]:
    mapping: Dict[str, Dict[str, List[float]]] = {}

    with open(path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for line in reader:
            if filter_list is not None:
                valid = True
                for condition in filter_list.items():
                    if str(line[condition[0]]) != condition[1]:
                        print(f"skipping (condition {condition[0]} is {line[condition[0]]} and not {condition[1]})")
                        valid = False
                        break
                if not valid:
                    continue
            algokey = " ".join([line[x] for x in x_main])
            if algokey not in mapping:
                mapping[algokey] = {}
            if x_groupby is not None:
                if line[x_groupby] not in mapping[algokey]:
                    mapping[algokey][str(line[x_groupby])] = [] # type: ignore
                try:
                    ans = float(line[y])
                    mapping[algokey][str(line[x_groupby])].append(ans) # type: ignore
                except ValueError:
                    continue
            else:
                if " " not in mapping[algokey]:
                    mapping[algokey][" "] = [] # type: ignore
                mapping[algokey][" "].append(float(line[y])) # type: ignore
    return mapping

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerar gráficos sin entrenar")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results/kohonen")
    args = parser.parse_args()


    plot_results(args.results_dir)
