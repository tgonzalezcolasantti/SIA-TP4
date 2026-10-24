"""Adaptado del esquema de configuración y carga de ej2/utils.py del TP3."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    # Las rutas del JSON se resuelven respecto de ese archivo.
    for key in ("dataset", "out_dir"):
        config[key] = str((path.parent / config[key]).resolve())
    return config


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def prepare_creators(path, features=None, exclude_columns=None):
    """Separa identificadores y ajusta z-score con desvío poblacional (ddof=0)."""
    df = pd.read_csv(path)
    identifiers = ["creator_id", "name"]
    if not set(identifiers).issubset(df.columns):
        raise ValueError("El CSV debe contener creator_id y name")
    if df.empty or df[identifiers].isna().any().any() or df.creator_id.duplicated().any():
        raise ValueError("Se requieren creadores con identificadores completos y únicos")
    if features is None:
        excluded = set(identifiers) | set(exclude_columns or [])
        unknown = excluded - set(df.columns)
        if unknown:
            raise ValueError(f"Columnas excluidas inexistentes: {sorted(unknown)}")
        features = [c for c in df.columns if c not in excluded]
    if not features or len(set(features)) != len(features) or set(features) & set(identifiers):
        raise ValueError("Las variables deben ser únicas y no incluir identificadores")
    numeric = df[features].apply(pd.to_numeric, errors="raise")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Hay valores faltantes o infinitos; resolverlos en el EDA antes de entrenar")
    means, scales = values.mean(axis=0), values.std(axis=0)
    constants = [name for name, scale in zip(features, scales) if scale == 0]
    scales[scales == 0] = 1.0
    metadata = {
        "features": list(features), "mean": means.tolist(), "scale": scales.tolist(),
        "constant_features": constants, "ddof": 0,
    }
    return df, (values - means) / scales, metadata


def transform_features(df, metadata):
    """Aplica el mismo escalado a nuevos creadores, sin volver a ajustarlo."""
    values = df[metadata["features"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Los datos deben ser numéricos y finitos")
    return (values - np.asarray(metadata["mean"])) / np.asarray(metadata["scale"])
