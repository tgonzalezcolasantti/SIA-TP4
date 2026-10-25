from math import sqrt
from numbers import Number
from typing import Callable, List, Self, Tuple

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
        self.input_weights = np.empty((grid_size**2, input_data.shape[1]))
        indices = np.random.randint(0, len(input_data), size=grid_size**2)
        self.input_weights[:] = input_data[indices]
        self.train(input_data)

    def similarity(self: Self, input_data: np.ndarray) -> int:
        'Returns the index of the neuron with closest weights to this input'
        distances = np.linalg.norm(self.input_weights - input_data[np.newaxis, :], axis=1)
        return int(np.argmin(distances))

    def grid_position(self: Self, neuron: int) -> Tuple[int, int]:
        return neuron % self.grid_size, int(neuron / self.grid_size)

    def grid_distance(self: Self, n1: Tuple[int, int], n2: Tuple[int, int]) -> float:
        return sqrt((n1[0] - n2[0]) **2 + (n1[1] - n2[1]) **2)

    def grid_neighbors(self: Self, neuron: int, iteration: int) -> List[int]:
        neuron_position = self.grid_position(neuron)
        neighbors: List[int] = []
        for i in range(0, self.grid_size**2):
            if i != neuron and self.grid_distance(self.grid_position(i), neuron_position) <= self.radius(iteration):
                neighbors.append(i)
        return neighbors

    def train_single_input(self: Self, input_data: np.ndarray, iteration: int):
        winning_neuron = self.similarity(input_data)
        neighbors = self.grid_neighbors(winning_neuron, iteration)
        self.input_weights[neighbors] = self.input_weights[neighbors] + self.eta(iteration) * (input_data - self.input_weights[neighbors])

    def train(self: Self, input_data: np.ndarray):
        for iteration in range(self.iterations):
            print(f"Iteration {iteration}")
            for single_data in input_data:
                self.train_single_input(single_data, iteration)

