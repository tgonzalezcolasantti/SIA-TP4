from math import sqrt
from typing import List, Self, Tuple

import numpy as np

from models.neural_network import Perceptron


class Kohonnen():
    def __init__(self: Self, input_data: np.ndarray, grid_size: int, radius: float, learning_rate: float, iterations: int = 500):
        if input_data.ndim != 2:
            raise AttributeError("Input must be 2-dimensional")
        if radius <= 0 or radius > grid_size:
            raise AttributeError("Radius must be between 0 and grid size.")
        if learning_rate > 1 or learning_rate <=0:
            raise AttributeError("Learning rate must be between 0 and 1")

        self.radius = radius
        self.learning_rate = learning_rate
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

    def grid_neighbors(self: Self, neuron: int) -> List[int]:
        neuron_position = self.grid_position(neuron)
        neighbors: List[int] = []
        for i in range(0, self.grid_size**2):
            if i != neuron and self.grid_distance(self.grid_position(i), neuron_position) <= self.radius:
                neighbors.append(i)
        return neighbors

    def train_single_input(self: Self, input_data: np.ndarray, iteration: int):
        winning_neuron = self.similarity(input_data)
        neighbors = self.grid_neighbors(winning_neuron)
        self.input_weights[neighbors] = self.input_weights[neighbors] + self.learning_rate * (input_data - self.input_weights[neighbors])

    def train(self: Self, input_data: np.ndarray):
        for iteration in range(self.iterations):
            print(f"Iteration {iteration}")
            for single_data in input_data:
                self.train_single_input(single_data, iteration)