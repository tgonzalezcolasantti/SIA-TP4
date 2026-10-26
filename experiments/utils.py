"""Adaptado del esquema de configuración y carga de ej2/utils.py del TP3."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, TypedDict

import numpy as np
import pandas as pd

from models.standarization import ZScoreStandardize

ROOT = Path(__file__).resolve().parents[1]

class CreatorMetadata(TypedDict):
    features: List[str]
    mean: List[float]
    scale: List[float]
    constant_features: List[str]
    ddof: int

class KohonenConfig(TypedDict):
    dataset: str
    out_dir: str
    features: List[str]
    exclude_columns: List[str]
    id_columns: List[str]
    seeds: List[int]
    grid_sizes: List[int]
    radii: List[float]
    learning_rates: List[float]
    epochs: List[int]
    max_tasks: int

def load_config(path: Path) -> Dict[str, Any]:
    path = path.resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    # Las rutas del JSON se resuelven respecto de ese archivo.
    for key in ("dataset", "out_dir"):
        config[key] = str((path.parent / config[key]).resolve())
    return config


def save_json(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def prepare_creators(
    path: Path,
    identifiers: List[str],
    features: List[str],
    exclude_columns: List[str]
) -> Tuple[pd.DataFrame, ZScoreStandardize, CreatorMetadata]:
    """Separa identificadores y ajusta z-score con desvío poblacional (ddof=0)."""
    print("Loading creators")
    df = pd.read_csv(path)
    if not set(identifiers).issubset(df.columns):
        raise ValueError("El CSV debe contener creator_id y name")
    if df.empty or df[identifiers].isna().any().any() or df.creator_id.duplicated().any():
        raise ValueError("Se requieren creadores con identificadores completos y únicos")

    if not features:
        excluded = set(identifiers + exclude_columns)
        unknown = excluded - set(df.columns)
        if unknown:
            raise ValueError(f"Columnas excluidas inexistentes: {sorted(unknown)}")
        features = [c for c in df.columns if c not in excluded]
    elif len(set(features)) != len(features) or set(features) & set(identifiers):
        raise ValueError("Las variables deben ser únicas y no incluir identificadores")

    numeric = df[features].apply(pd.to_numeric, errors="raise")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Hay valores faltantes o infinitos; resolverlos en el EDA antes de entrenar")

    print("Standarizing creators")
    standardized = ZScoreStandardize(values)
    metadata: CreatorMetadata = {
        "features": list(features), "mean": standardized.means, "scale": standardized.scales,
        "constant_features": [features[i] for i in standardized.constants], "ddof": 0,
    }
    return df, standardized, metadata