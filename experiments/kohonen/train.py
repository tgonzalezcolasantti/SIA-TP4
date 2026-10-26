import argparse
import json
from pathlib import Path
import time

import numpy as np


from experiments.utils import KohonenConfig
from experiments.utils import load_config, prepare_creators
from models.standarization import ZScoreStandardize
from models.unsupervised import Kohonen

def train_model(data: ZScoreStandardize, grid: int, radius: float, eta: float, epochs: int, seed: int) -> Kohonen:
    np.random.seed(seed)
    return Kohonen(data, grid, radius, eta, epochs)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--grid", type=int)
    parser.add_argument("--radius", type=float)
    parser.add_argument("--eta", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--label", type=str)
    args = parser.parse_args()

    config: KohonenConfig = load_config(args.config) # type: ignore

    _, data, _ = prepare_creators(
        Path(config["dataset"]), config['id_columns'], config["features"], config["exclude_columns"]
    )
    start = time.perf_counter()
    kohonen = train_model(data, args.grid, args.radius, args.eta, args.epochs, args.seed)

    print(json.dumps({"config_id": args.label, "grid_size": args.grid, "radius": args.radius,
                         "learning_rate": args.eta, "epochs": args.epochs, "seed": args.seed,
                         **kohonen.metrics, "seconds": time.perf_counter() - start}))


if __name__ == '__main__':
    main()