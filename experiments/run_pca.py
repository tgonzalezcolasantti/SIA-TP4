import argparse
from pathlib import Path
from typing import List, TypedDict
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
    _, data, preprocessing = prepare_creators(
        config["dataset"], config["id_columns"], config["features"], config["exclude_columns"]
    )
    out = Path(config["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    print("Running PCA analysis")
    pca = PCA(n_components = config["components"]).fit(data.standard_values)

    pd.DataFrame(pca.components_).to_csv(out / "eigenvectors.csv", index=False, header = preprocessing["features"])
    eigenvalues = np.asarray([pca.explained_variance_, pca.explained_variance_ratio_, np.cumsum(pca.explained_variance_ratio_)]).T
    pd.DataFrame(eigenvalues).to_csv(out / "eigenvalues.csv", index = False, header = ["Eigenvalue", "Variance ratio", "Cumulative variance ratio"])
    return out

def plot_results(out):
    pass

def main():
    parser = argparse.ArgumentParser(description="Realizar PCA sobre los creadores del TP4")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/PCA.json")
    args = parser.parse_args()
    out = run_study(load_config(args.config)) # type: ignore
    plot_results(out)
    print(f"Resultados y gráficos: {out}")


if __name__ == "__main__":
    main()
