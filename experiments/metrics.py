"""Métricas propias del SOM: no requieren etiquetas supervisadas."""

import numpy as np


def evaluate_som(model, data):
    distances = np.linalg.norm(data[:, None, :] - model.input_weights[None, :, :], axis=2)
    closest = np.argsort(distances, axis=1, kind="stable")[:, :2]
    bmus = closest[:, 0]
    # Vecindad de cuatro conexiones (sin diagonales) para error topográfico.
    coords = np.array([model.grid_position(i) for i in range(model.grid_size**2)])[closest]
    adjacent = np.abs(coords[:, 0] - coords[:, 1]).sum(axis=1) == 1
    hits = np.bincount(bmus, minlength=model.grid_size**2)
    return {
        "quantization_error": float(distances[np.arange(len(data)), bmus].mean()),
        "topographic_error": float(np.mean(~adjacent)),
        "empty_neurons": int(np.sum(hits == 0)),
        "occupancy": float(np.mean(hits > 0)),
    }, bmus


def u_matrix(model):
    """Distancia media a vecinos ortogonales; misma topología que la métrica."""
    result = np.zeros(model.grid_size**2)
    coordinates = np.array([model.grid_position(i) for i in range(model.grid_size**2)])
    for i, position in enumerate(coordinates):
        neighbors = np.abs(coordinates - position).sum(axis=1) == 1
        result[i] = np.linalg.norm(model.input_weights[neighbors] - model.input_weights[i], axis=1).mean()
    # grid_position devuelve (x, y); reshape organiza filas y columnas.
    return result.reshape(model.grid_size, model.grid_size)
