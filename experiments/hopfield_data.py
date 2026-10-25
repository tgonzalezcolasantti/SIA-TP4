"""Patrones y deterioro bipolar; adaptación del experimento de ruido del TP3."""

from pathlib import Path

import matplotlib.image as mpimg
import numpy as np


LETTERS = {
    "A": [".***.", "*...*", "*****", "*...*", "*...*"],
    "K": ["*...*", "*..*.", "***..", "*..*.", "*...*"],
    "S": [".****", "*....", ".***.", "....*", "****."],
    "J": ["*****", "...*.", "...*.", "*..*.", ".**.."],
    "L": ["*....", "*....", "*....", "*....", "*****"],
    "I": ["*****", "..*..", "..*..", "..*..", "*****"],
    "O": [".***.", "*...*", "*...*", "*...*", ".***."],
    "X": ["*...*", ".*.*.", "..*..", ".*.*.", "*...*"],
}


def letter_patterns(names):
    if not names or len(set(names)) != len(names) or set(names) - LETTERS.keys():
        raise ValueError(f"Elegir letras distintas de {list(LETTERS)}")
    return np.array([[1 if pixel == "*" else -1 for row in LETTERS[name] for pixel in row]
                     for name in names], dtype=np.int8)


def load_qr_patterns(directory):
    paths = sorted(Path(directory).glob("*.png"))
    if len(paths) != 5:
        raise ValueError("Se requieren exactamente los cinco QR originales en PNG")
    patterns = []
    for path in paths:
        pixels = mpimg.imread(path)
        if pixels.shape != (37, 37):
            raise ValueError(f"{path.name}: se esperaba una imagen gris de 37 x 37")
        if not np.isfinite(pixels).all() or not np.isin(pixels, [0, 1]).all():
            raise ValueError(f"{path.name}: se esperaba un PNG binario, sin redimensionar")
        # Convención: negro = +1, blanco = -1.
        patterns.append(np.where(pixels < 0.5, 1, -1).ravel())
    return [p.stem for p in paths], np.array(patterns, dtype=np.int8), paths


def add_noise(pattern, fraction, rng):
    """Invierte floor(fraction*N) posiciones distintas: x -> -x (no 1-x)."""
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("El nivel de ruido debe estar entre 0 y 1")
    noisy = np.asarray(pattern).copy()
    if not noisy.size or not np.isin(noisy, [-1, 1]).all():
        raise ValueError("El patrón debe ser bipolar y no vacío")
    flat = noisy.reshape(-1)
    indices = rng.choice(flat.size, int(fraction * flat.size), replace=False)
    flat[indices] *= -1
    return noisy


def erase_corner(pattern, fraction):
    """Simula una esquina superior izquierda blanca de lado floor(fraction*lado)."""
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("La fracción del lado debe estar entre 0 y 1")
    side = int(np.sqrt(np.asarray(pattern).size))
    if side * side != np.asarray(pattern).size:
        raise ValueError("Se requiere un patrón cuadrado")
    result = np.asarray(pattern).copy().reshape(side, side)
    width = int(side * fraction)
    result[:width, :width] = -1
    return result.reshape(np.asarray(pattern).shape)


def normalized_overlap(patterns):
    values = np.asarray(patterns, dtype=float)
    return values @ values.T / values.shape[1]
