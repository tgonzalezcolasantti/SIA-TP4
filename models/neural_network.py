from pathlib import Path
import pickle
from typing import List, Optional, Self, Tuple, override

import numpy as np

from models.optimization import NoOptimization, Optimization

from .activation import Activation


class Perceptron:
    def __init__(
        self: Self,
        neuron_topology: List[int],
        activation: Activation,
        learning_rate: float,
        save: Path | None = None,
        epoch: int = 0,
        weights: np.ndarray | None = None,
        optimization: Optimization = NoOptimization(),
        model_name: Optional[str] = None,
    ):
        self.model_name = model_name
        self.topology = neuron_topology
        self.learning_rate = learning_rate
        self.activation = activation
        self.optimization = optimization
        self.epoch = epoch
        self.error_history = []

        if save:
            if save.is_dir():
                save = save / self._suggested_filename()
            if save.exists():
                unpickled: Perceptron = self.load(save)
                weights = unpickled.weights
                neuron_topology = unpickled.topology
                self.activation = unpickled.activation
                self.optimization = unpickled.optimization
                self.epoch = unpickled.epoch
                self.learning_rate = unpickled.learning_rate
                self.error_history = unpickled.error_history
        else:
            if len(neuron_topology) < 2 or any(count <= 0 for count in neuron_topology):
                raise ValueError(
                    "neuron_topology must contain at least two positive layer sizes"
                )

        total = sum(neuron_topology)
        self.weights = np.zeros((total, total)) if weights is None else weights
        self.outputs = np.zeros(total)
        self.h = np.zeros(total)
        self.deltas = np.zeros(total)
        self.gradient = np.zeros(self.weights.shape)
        self.last_layer_count = neuron_topology[-1]
        self.save_path = save
        boundaries = np.cumsum([0, *neuron_topology])
        self.layer_slices = [
            slice(boundaries[i], boundaries[i + 1]) for i in range(len(neuron_topology))
        ]
        if weights is None:
            self._init_weights(neuron_topology)

    def _init_weights(self: Self, neuron_topology: List[int]):
        for layer_idx in range(1, len(neuron_topology)):
            previous = self.layer_slices[layer_idx - 1]
            current = self.layer_slices[layer_idx]
            # TODO What is this? aka why this function specifically?
            limit = np.sqrt(
                6 / (neuron_topology[layer_idx - 1] + neuron_topology[layer_idx])
            )
            self.weights[current, previous] = np.random.uniform(
                -limit,
                limit,
                (neuron_topology[layer_idx], neuron_topology[layer_idx - 1]),
            )

    def classify(self: Self, inputs: np.ndarray) -> np.ndarray:
        inputs = np.asarray(inputs, dtype=float)
        if inputs.ndim == 1:
            inputs = inputs[None, :]
        if inputs.ndim != 2 or inputs.shape[1] != self.topology[0]:
            raise ValueError("input size does not match the first layer")
        return self._calculate_outputs(inputs).T.copy()

    def _calculate_outputs(self: Self, inputs: np.ndarray) -> np.ndarray:
        inputs = np.asarray(inputs, dtype=float)
        if inputs.ndim == 1:
            inputs = inputs[None, :]
        shape = (self.weights.shape[0], len(inputs))
        if self.outputs.shape != shape:
            self.outputs = np.zeros(shape)
            self.h = np.zeros(shape)
        self.outputs[self.layer_slices[0]] = inputs.T
        for layer_idx in range(1, len(self.layer_slices)):
            previous = self.layer_slices[layer_idx - 1]
            current = self.layer_slices[layer_idx]
            bias = np.diagonal(self.weights[current, current])[:, None]
            self.h[current] = self.weights[current, previous] @ self.outputs[previous] + bias
            self.outputs[current] = self.activation.excite(self.h[current])
        return self.outputs[self.layer_slices[-1]]

    def _error(self: Self, output_errors: np.ndarray) -> float:
        return np.sum(output_errors ** 2) / 2

    def _global_errors(
        self: Self, results: np.ndarray, expected: np.ndarray
    ) -> Tuple[float, float]:
        errors = results - expected
        mse = np.mean(errors**2)
        mae = np.mean(np.abs(errors))
        print(f"MSE: {mse}\tMAE: {mae}")
        return mse, mae

    def _update_weights(self: Self, output_errors: np.ndarray):
        batch_count = output_errors.shape[1]
        output_layer = self.layer_slices[-1]
        output_derivatives = self.activation.derivative(self.h[output_layer])
        if self.deltas.shape != (self.weights.shape[0], output_errors.shape[1]):
            self.deltas = np.zeros((self.weights.shape[0], output_errors.shape[1]))
        else:
            self.deltas[:] = 0
        self.deltas[output_layer] = output_errors * output_derivatives

        #backpropagation for derivatives
        for layer_idx in range(len(self.layer_slices) - 2, 0, -1):
            current = self.layer_slices[layer_idx]
            following = self.layer_slices[layer_idx + 1]
            derivatives = self.activation.derivative(self.h[current])
            self.deltas[current] = (
                self.weights[following, current].T @ self.deltas[following]
            ) * derivatives

        for layer_idx in range(1, len(self.layer_slices)):
            previous = self.layer_slices[layer_idx - 1]
            current = self.layer_slices[layer_idx]
            self.gradient[current, previous] = self.deltas[current] @ self.outputs[previous].T / batch_count
        # gradient = self.deltas @ self.outputs.T / batch_count
        self.gradient[range(len(self.weights)), range(len(self.weights))] = np.sum(self.deltas, axis=1) / output_errors.shape[1]
        self.weights += self.optimization.apply(
            self.learning_rate, self.gradient
        )

    def train(
        self: Self,
        training_data: List[Tuple[np.ndarray, np.ndarray]],
        epochs: int,
        epsilon: float,
        batch_size: int = 1,
        with_history: bool = False,
        seed: int | None = None,
    ) -> List[Tuple[float, float]]:
        if epochs < 0 or batch_size == 0 or batch_size < -1:
            raise ValueError("epochs must be nonnegative and batch size positive or -1")
        if not training_data:
            raise ValueError("training_data must not be empty")
        inputs = np.asarray([sample[0] for sample in training_data], dtype=float)
        expected = np.asarray([sample[1] for sample in training_data], dtype=float)
        if expected.ndim == 1:
            expected = expected[:, None]
        if expected.shape != (len(inputs), self.last_layer_count):
            raise ValueError("expected output shape does not match the last layer")
        effective_batch_size = len(inputs) if batch_size == -1 else batch_size
        rng = np.random.default_rng(seed)
        for i in range(self.epoch, epochs):
            print(f"Epoch {i}", flush=True)
            order = rng.permutation(len(inputs)) if seed is not None else np.arange(len(inputs))
            for start in range(0, len(inputs), effective_batch_size):
                indices = order[start:start + effective_batch_size]
                outputs = self._calculate_outputs(inputs[indices])
                self._update_weights(outputs - expected[indices].T)
            self.epoch += 1
            if with_history or epsilon >= 0:
                predictions = self.classify(inputs)
                metrics = self._global_errors(predictions, expected)
                if with_history:
                    self.error_history.append(metrics)
                if epsilon >= 0 and metrics[0] / 2 <= epsilon:
                    self.save()
                    return self.error_history
            self.save()
        return self.error_history

    @classmethod
    def load(cls:type[Perceptron], save_path: Path):
        with open(save_path, 'rb+') as file:
            return pickle.load(file)

    def save(self: Self):
        if self.save_path:
            with open(self.save_path, '+wb') as file:
                pickle.dump(self, file)

    def _suggested_filename(self: Self) -> str:
        return f"{self.model_name + ' ' if self.model_name else None}"+\
               f"[{','.join(str(x) for x in self.topology)}]"+\
               f" {self.learning_rate:.6g} {str(self.activation)} {str(self.optimization)}.model"


class SimplePerceptron(Perceptron):
    def __init__(
        self: Self,
        inputs: int,
        activation: Activation,
        learning_rate: float,
        weight_range: Tuple[float, float] = (0, 1),
        epoch: int = 0,
        save: Optional[Path] = None,
        model_name: Optional[str] = None
    ):
        self.min_weight = min(weight_range)
        self.max_weight = max(weight_range)
        super().__init__(
            neuron_topology=[inputs, 1],
            activation=activation,
            learning_rate=learning_rate,
            save=save,
            model_name=model_name,
            epoch=epoch
        )

    @override
    def _init_weights(self: Self, neuron_topology: List[int]):
        for layer_idx in range(1, len(neuron_topology)):
            previous = self.layer_slices[layer_idx - 1]
            current = self.layer_slices[layer_idx]
            self.weights[current, previous] = np.random.uniform(
                self.min_weight,
                self.max_weight,
                (neuron_topology[layer_idx], neuron_topology[layer_idx - 1]),
            )


class MultiLayerPerceptron(Perceptron):
    pass
