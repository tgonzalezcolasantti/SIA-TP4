import argparse
from pathlib import Path
from typing import List, TypedDict, Tuple
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from experiments.utils import ROOT, load_config, prepare_creators

class PCAConfig(TypedDict):
    dataset: Path
    out_dir: Path
    id_columns: List[str]
    features: List[str]
    exclude_columns: List[str]
    components: int

def run_study(config: PCAConfig) -> Path:
    df, data, preprocessing = prepare_creators(
        config["dataset"], config["id_columns"], config["features"], config["exclude_columns"]
    )
    out = Path(config["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    print("Running PCA analysis")
    pca = PCA(n_components = config["components"]).fit(data.standard_values)

    pd.DataFrame(pca.components_).to_csv(out / "eigenvectors.csv", index=False, header = preprocessing["features"])
    eigenvalues = np.asarray([pca.explained_variance_, pca.explained_variance_ratio_, np.cumsum(pca.explained_variance_ratio_)]).T
    pd.DataFrame(eigenvalues).to_csv(out / "eigenvalues.csv", index = False, header = ["Eigenvalue", "Variance ratio", "Cumulative variance ratio"])
    scores = []
    for idx, score in enumerate(pca.transform(data.standard_values)):
        row = {}
        for label in config["id_columns"]:
            row[label] = df[label].iloc[idx]
        for pc, val in enumerate(score):
            row[f"PC{pc+1}"] = val
        scores.append(row)
    pd.DataFrame(scores).to_csv(out / "scores.csv", index = False)
    return out

def save_figure(fig: Figure, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_biplot(out: Path):
    pc_df = pd.read_csv(out / "eigenvectors.csv")
    pc_data = pc_df.to_numpy()[:2]
    names = pc_df.columns
    variance_data = pd.read_csv(out / "eigenvalues.csv").to_numpy()[:2,1]
    loadings = pc_data.T
    scores = pd.read_csv(out / "scores.csv")

    coso: Tuple[Figure, List[Axes]] = plt.subplots(1, 2, figsize=(20, 8))
    fig, axes = coso
    axes[1].scatter(scores["PC1"], scores["PC2"], alpha=0.75)
    axes[1].set_title("Biplot scatter")
    
    def horizontalalignment(lx):
        if abs(lx) < 0.1:
            return "center"
        if lx > 0:
            return "left"
        return "right"
    def verticalalignment(ly):
        if abs(ly) < 0.1:
            return "center"
        if ly > 0:
            return "top"
        return "bottom"

    for name, (lx, ly) in zip(names, loadings):
        axes[0].arrow(0, 0, lx, ly, color="r", alpha=0.5)
        axes[0].text(lx +  0.02 * np.sign(lx), ly + 0.02 * np.sign(ly), name, color="g", horizontalalignment=horizontalalignment(lx), verticalalignment=verticalalignment(ly))
    axes[0].set_title("Biplot composicion de PC1 y PC2")
    axes[0].set_xlim(min(loadings.flatten())*2, max(loadings.flatten())*2)
    axes[0].set_ylim(min(loadings.flatten())*2, max(loadings.flatten())*2)

    for ax in axes:
        ax.axhline(0, color="gray", lw=0.8)
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_xlabel(f"PC1 ({variance_data[0]:.1%})")
        ax.set_ylabel(f"PC2 ({variance_data[1]:.1%})")
    save_figure(fig, out / "biplot.png")

def plot_index(out):
    scores = pd.read_csv(out / "scores.csv")

    fig, ax = plt.subplots(figsize=(10,5))

    scores = scores.sort_values(axis=0, by="PC1")

    ax.bar(np.arange(len(scores)), scores["PC1"])
    ax.set_xticks(np.arange(len(scores)),scores["name"], rotation=90)
    ax.set_ylabel("PC1")
    ax.set_title("PC1 por influencer (ordenado)")
    save_figure(fig, out / "index.png")

def plot_results(out):
    plot_biplot(out)
    plot_index(out)

def main():
    parser = argparse.ArgumentParser(description="Realizar PCA sobre los creadores del TP4")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/PCA.json")
    args = parser.parse_args()
    out = run_study(load_config(args.config)) # type: ignore
    plot_results(out)
    print(f"Resultados y gráficos: {out}")


if __name__ == "__main__":
    main()
