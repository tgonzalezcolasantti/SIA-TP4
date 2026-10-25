"""Visualizaciones de patrones, recuperación, capacidad y energía de Hopfield."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def pattern_image(pattern, path, side):
    # PNG de exactamente side x side, sin interpolación ni decoración.
    plt.imsave(path, np.asarray(pattern).reshape(side, side), cmap="gray_r", vmin=-1, vmax=1)


def recovery_figure(original, query, result, side, path, title):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
    for ax, pattern, label in zip(axes[:3], [original, query, result.state],
                                 ["Original", "Consulta deteriorada", "Recuperado"]):
        ax.imshow(pattern.reshape(side, side), cmap="gray_r", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(label)
        ax.axis("off")
    axes[3].plot(range(len(result.energies)), result.energies, marker=".")
    axes[3].set_xlabel("Barridos completos")
    axes[3].set_ylabel("Energía")
    axes[3].set_title("Convergió" if result.converged else "Límite alcanzado")
    fig.suptitle(title, fontsize=10)
    save_figure(fig, path)


def overview(out, names, patterns, side):
    out = Path(out)
    fig, axes = plt.subplots(1, len(names), figsize=(3 * len(names), 3), squeeze=False)
    for ax, name, pattern in zip(axes.flat, names, patterns):
        ax.imshow(pattern.reshape(side, side), cmap="gray_r", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(name.replace("_menu_qr", ""), fontsize=9)
        ax.axis("off")
    save_figure(fig, out / "stored_patterns.png")
    overlap = pd.read_csv(out / "overlap.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(overlap, vmin=-1, vmax=1, cmap="coolwarm")
    labels = [n.replace("_menu_qr", "") for n in names]
    ax.set_xticks(range(len(names)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), labels)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{overlap.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    ax.set_title("Solapamiento normalizado: producto interno / N")
    save_figure(fig, out / "overlap.png")


def summary_figures(out):
    out = Path(out)
    summary = pd.read_csv(out / "summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for damage, group in summary.groupby("damage"):
        for ax, metric, label in zip(axes, ["success_rate", "sweeps_mean", "hamming_final_mean"],
                                     ["Recuperación exacta", "Barridos medios", "Distancia de Hamming final"]):
            ax.plot(group.level, group[metric], "o-", label=damage)
            ax.set_title(label)
            ax.set_xlabel("Fracción de bits invertidos / lado borrado")
            ax.grid(alpha=0.3)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend()
    save_figure(fig, out / "summary.png")


def capacity_figure(out):
    out = Path(out)
    frame = pd.read_csv(out / "capacity_summary.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(frame.stored_count, frame.success_rate, "o-")
    ax.set_xticks(frame.stored_count)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Cantidad de letras almacenadas")
    ax.set_ylabel("Fracción de recuperaciones exactas")
    ax.set_title("Capacidad empírica para este orden de letras y ruido configurado")
    ax.grid(alpha=0.3)
    save_figure(fig, out / "capacity.png")
