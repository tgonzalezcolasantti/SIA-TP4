"""Memoria asociativa bipolar con Hebb y recuperación asincrónica."""

from dataclasses import dataclass

import numpy as np


@dataclass
class RecallResult:
    state: np.ndarray
    converged: bool
    sweeps: int
    energies: list[float]
    changes: list[int]
    history: list[np.ndarray]
    update_energies: list[float]


class Hopfield:
    def __init__(self):
        self.weights = None
        self.patterns = None

    @staticmethod
    def _bipolar(values):
        array = np.asarray(values)
        if not array.size or not np.isin(array, [-1, 1]).all():
            raise ValueError("Los patrones deben ser no vacíos y contener solamente -1 y 1")
        return array.astype(np.float64)

    def train(self, patterns):
        patterns = self._bipolar(patterns)
        if patterns.ndim != 2:
            raise ValueError("Se requiere una matriz (cantidad de patrones, neuronas)")
        self.patterns = patterns.copy()
        # Float evita overflow en productos de vectores int8 de tamaño 1369.
        self.weights = patterns.T @ patterns / patterns.shape[1]
        np.fill_diagonal(self.weights, 0)
        return self

    def _state(self, pattern):
        if self.weights is None:
            raise RuntimeError("Entrenar la red antes de usarla")
        state = self._bipolar(pattern).reshape(-1)
        if len(state) != len(self.weights):
            raise ValueError("La consulta no tiene la dimensión de los patrones almacenados")
        return state

    def energy(self, pattern):
        state = self._state(pattern)
        return float(-0.5 * state @ self.weights @ state)

    def recall(self, pattern, max_sweeps=50, seed=None, record_history=False, record_updates=False):
        if isinstance(max_sweeps, bool) or not isinstance(max_sweeps, int) or max_sweeps < 1:
            raise ValueError("max_sweeps debe ser un entero positivo")
        state = self._state(pattern).copy()
        rng = np.random.default_rng(seed)
        field = self.weights @ state
        energy = float(-0.5 * state @ field)
        energies = [energy]
        changes = []
        history = [state.astype(np.int8)] if record_history else []
        update_energies = [energy] if record_updates else []
        converged = False
        for sweep in range(1, max_sweeps + 1):
            changed = 0
            # Cada neurona se visita exactamente una vez por barrido.
            for i in rng.permutation(len(state)):
                h = field[i]
                # Ante campo nulo se conserva el valor; tolerancia de redondeo.
                new = 1 if h > 1e-12 else -1 if h < -1e-12 else state[i]
                delta = new - state[i]
                if delta:
                    energy -= float(delta * h)
                    state[i] = new
                    field += delta * self.weights[:, i]
                    changed += 1
                if record_updates:
                    update_energies.append(energy)
            energies.append(energy)
            changes.append(changed)
            if record_history:
                history.append(state.astype(np.int8))
            if changed == 0:
                converged = True
                break
        return RecallResult(state.astype(np.int8), converged, sweep, energies, changes, history, update_energies)


def classify_recall(result, stored, target=None):
    """Un estado espurio debe ser estable; se distinguen patrones invertidos."""
    if not result.converged:
        return "not_converged"
    stored = np.asarray(stored)
    matches = np.flatnonzero(np.all(stored == result.state, axis=1))
    if len(matches):
        return "correct" if target is None or target in matches else "other_memory"
    if np.any(np.all(-stored == result.state, axis=1)):
        return "inverse_memory"
    return "spurious"
