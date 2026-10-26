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
from models.unsupervised import Oja

class PCAConfig(TypedDict):
    dataset: Path
    out_dir: Path
    id_columns: List[str]
    features: List[str]
    exclude_columns: List[str]
    components: int
    oja_epochs: int
    oja_eta: float

def dump_scores(out: Path, df: pd.DataFrame, id_cols: List[str], scores: np.ndarray):
    score_dicts = []
    for idx, score in enumerate(scores):
        row = {}
        for label in id_cols:
            row[label] = df[label].iloc[idx]
        for pc, val in enumerate(score):
            row[f"PC{pc+1}"] = val
        score_dicts.append(row)
    pd.DataFrame(score_dicts).to_csv(out, index = False)

def run_study(config: PCAConfig) -> Path:
    df, data, preprocessing = prepare_creators(
        config["dataset"], config["id_columns"], config["features"], config["exclude_columns"]
    )
    out = Path(config["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    print("Running PCA analysis")
    pca = PCA(n_components = config["components"]).fit(data.standard_values)

    pd.DataFrame(pca.components_).to_csv(
        out / "eigenvectors.csv",
        index=False,
        header = preprocessing["features"]
    )
    eigenvalues = np.asarray([
        pca.explained_variance_,
        pca.explained_variance_ratio_,
        np.cumsum(pca.explained_variance_ratio_)
    ]).T
    pd.DataFrame(eigenvalues).to_csv(
        out / "eigenvalues.csv",
        index = False,
        header = ["Eigenvalue (Variance)", "Variance ratio", "Cumulative variance ratio"]
    )
    dump_scores(out / "scores.csv", df, config["id_columns"], pca.transform(data.standard_values))

    print("Running Oja")
    oja_coefficients = Oja(
        data.standard_values,
        config["oja_epochs"],
        lambda x: config["oja_eta"]
    ).reshape((1,-1))
    pd.DataFrame(oja_coefficients).to_csv(
        out / "oja.csv",
        index=False,
        header = preprocessing["features"]
    )
    dump_scores(
        out / "scores_oja.csv",
        df,
        config["id_columns"],
        np.asarray(np.inner(data.standard_values, oja_coefficients.flatten())).reshape((-1,1))
    )
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
    for index, entry in scores.iterrows():
        axes[1].text(entry["PC1"] + np.random.rand()*.2, entry["PC2"] + np.random.rand()*.2, entry["name"])
    axes[1].set_title("Biplot scatter")

    def horizontal_alignment(lx):
        if abs(lx) < 0.1:
            return "center"
        if lx > 0:
            return "left"
        return "right"
    def vertical_alignment(ly):
        if abs(ly) < 0.1:
            return "center"
        if ly > 0:
            return "top"
        return "bottom"

    for name, (lx, ly) in zip(names, loadings):
        axes[0].arrow(0, 0, lx, ly, color="r", alpha=0.5)
        axes[0].text(
            lx +  0.02 * np.sign(lx),
            ly + 0.02 * np.sign(ly),
            name,
            color="g",
            horizontalalignment=horizontal_alignment(lx),
            verticalalignment=vertical_alignment(ly)
        )

    print(f"{'parametro':>30s}  angulo\tmodulo")
    for name, (lx, ly) in sorted(zip(names, loadings), key= lambda x: np.atan2(x[1][1], x[1][0])):
        print(f"{name:>30s} {np.atan2(ly, lx):.4f}\t{np.sqrt(lx**2+ly**2):.4f}")
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

def plot_pc1_weights(out):
    scores = pd.read_csv(out / "eigenvectors.csv")

    fig, ax = plt.subplots(figsize=(10,5))

    pc1 = scores.iloc[0]
    ax.bar(np.arange(len(pc1)), pc1)
    ax.set_xticks(np.arange(len(pc1)),scores.columns, rotation=90)
    ax.set_ylabel("Ponderacion")
    ax.set_title("Ponderacion de componentes de PC1")
    save_figure(fig, out / "pc1_composition.png")

def plot_oja_index_comparison(out):
    scores = pd.read_csv(out / "scores.csv")
    scores_oja = pd.read_csv(out / "scores_oja.csv")

    if all(scores_oja["PC1"] * scores["PC1"] <= 0.1): #Changui
        error = scores["PC1"] + scores_oja["PC1"]
    else:
        error = scores["PC1"] - scores_oja["PC1"]

    print(f"MSE (PC1 - OJA): {(error**2).mean():.4e}")

def plot_results(out):
    plot_biplot(out)
    plot_index(out)
    plot_pc1_weights(out)
    plot_oja_index_comparison(out)

def main():
    parser = argparse.ArgumentParser(description="Realizar PCA sobre los creadores del TP4")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/PCA.json")
    args = parser.parse_args()
    out = run_study(load_config(args.config)) # type: ignore
    plot_results(out)
    print(f"Resultados y gráficos: {out}")


if __name__ == "__main__":
    main()
