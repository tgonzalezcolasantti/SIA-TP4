from typing import Callable, Dict, Self

import numpy as np

class ConstantValue:
    def __init__(self: Self, value: float):
        self.value = value
    def __call__(self: Self, iteration: int) -> float:
        return self.value

class LinearValue:
    def __init__(self: Self, initial_value: float, minimum_value: float, target_iterations: int):
        self.initial_value = initial_value
        self.minimum_value = minimum_value
        self.target_iterations = target_iterations
    def __call__(self: Self, iteration: int) -> float:
        return max(self.minimum_value, (self.initial_value - self.minimum_value) * (self.target_iterations - iteration) + self.minimum_value)

class Kohonen():
    def __init__(
        self: Self,
        input_data: np.ndarray,
        grid_size: int,
        radius: Callable[[int], float] | float,   #radius(iteration)
        eta: Callable[[int], float] | float, #learning_rate(iteration)
        iterations: int = 500
    ):
        if input_data.ndim != 2:
            raise AttributeError("Input must be 2-dimensional")

        self.radius: Callable[[int], float] = ConstantValue(radius) if isinstance(radius, float) else radius # type: ignore
        self.eta: Callable[[int], float] = ConstantValue(eta) if isinstance(eta, float) else eta # type: ignore
        self.grid_size = grid_size
        self.iterations = iterations
        self.bmus: np.ndarray
        self.metrics: Dict[str, float | int]
        self.input_weights = np.empty((grid_size**2, input_data.shape[1]))
        indices = np.random.randint(0, len(input_data), size=grid_size**2)
        self.input_weights[:] = input_data[indices]
        self.grid_position = np.asarray(np.divmod(np.arange(self.grid_size**2), self.grid_size)).T
        self.grid_distance = np.empty((grid_size**2, grid_size**2))
        self._grid_distances()
        self._train(input_data)
        self._evaluate_som(input_data)


    def similarity(self: Self, input_data: np.ndarray) -> np.ndarray:
        return np.linalg.norm(self.input_weights - input_data[np.newaxis, :], axis=1)

    def most_similar(self: Self, input_data: np.ndarray) -> int:
        'Returns the index of the neuron with closest weights to this input'
        return int(np.argmin(self.similarity(input_data)))

    def _grid_distances(self: Self):
        neurons = np.arange(self.grid_size**2)
        for neuron in neurons:
            self.grid_distance[neuron] = np.sqrt(
                (self.grid_position[neurons,0] - self.grid_position[neuron,0]) **2 +
                (self.grid_position[neurons,1] - self.grid_position[neuron,1]) **2
            )

    def grid_neighbors(self: Self, neuron: int, iteration: int) -> np.ndarray:
        return self.grid_distance[neuron] <= self.radius(iteration)

    def train_single_input(self: Self, input_data: np.ndarray, iteration: int):
        winning_neuron = self.most_similar(input_data)
        neighbors = self.grid_neighbors(winning_neuron, iteration)
        self.input_weights[neighbors] = self.input_weights[neighbors] + self.eta(iteration) * (input_data - self.input_weights[neighbors])

    def _train(self: Self, input_data: np.ndarray):
        for iteration in range(self.iterations):
            # print(f"Iteration {iteration}")
            for single_data in input_data:
                self.train_single_input(single_data, iteration)

    def _evaluate_som(self: Self, data: np.ndarray):
        distances = np.linalg.norm(data[:, None, :] - self.input_weights[None, :, :], axis=2)
        closest = np.argsort(distances, axis=1, kind="stable")[:, :2]
        self.bmus = closest[:, 0]
        # Vecindad de cuatro conexiones (sin diagonales) para error topográfico.
        coords = np.array([self.grid_position[i] for i in range(self.grid_size**2)])[closest]
        adjacent = np.abs(coords[:, 0] - coords[:, 1]).sum(axis=1) == 1
        hits = np.bincount(self.bmus, minlength=self.grid_size**2)
        self.metrics = {
            "quantization_error": float(distances[np.arange(len(data)), self.bmus].mean()),
            "topographic_error": float(np.mean(~adjacent)),
            "empty_neurons": int(np.sum(hits == 0)),
            "occupancy": float(np.mean(hits > 0)),
        }

    def u_matrix(self: Self):
        """Distancia media a vecinos ortogonales; misma topología que la métrica."""
        result = np.zeros(self.grid_size**2)
        for i in range(self.grid_size**2):
            neighbors = self.grid_distance[i] == 1
            result[i] = np.linalg.norm(self.input_weights[neighbors] - self.input_weights[i], axis=1).mean()
        # grid_position devuelve (x, y); reshape organiza filas y columnas.
        return result.reshape(self.grid_size, self.grid_size)
