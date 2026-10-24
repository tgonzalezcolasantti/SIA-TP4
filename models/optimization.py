from abc import ABC, abstractmethod
from typing import Self, override

import numpy as np


class Optimization(ABC):
    name = ""
    @abstractmethod
    def apply(self: Self, learning_rate: float, gradient: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def __str__(self: Self)-> str:
        return self.name

    @classmethod
    def from_string(cls: type[Optimization], string: str) -> Optimization:
        for subclass in cls.__subclasses__():
            if subclass.name == string:
                return subclass()
        raise ValueError(f"{string} is not a valid optimizer name")

class NoOptimization(Optimization):
    name="None"
    @override
    def apply(self: Self, learning_rate: float, gradient: np.ndarray) -> np.ndarray:
        return -learning_rate * gradient

class Momentum(Optimization):
    name="Momentum"

    def __init__(self: Self, alpha: float = 0.9):
        super().__init__()
        self.alpha = alpha
        self.last_errors: np.ndarray | None = None

    @override
    def apply(self: Self, learning_rate: float, gradient: np.ndarray) -> np.ndarray:
        if self.last_errors is None:
            self.last_errors = np.zeros(gradient.shape)
        self.last_errors = -learning_rate * gradient + self.alpha*self.last_errors
        return self.last_errors

    def __str__(self: Self) -> str:
        return f"{self.name}({self.alpha:.6g})"

class RMSProp(Optimization):
    name="RMSProp"

    def __init__(self: Self, gamma: float = 0.1):
        super().__init__()
        self.gamma = gamma
        self.last: np.ndarray | None = None

    @override
    def apply(self: Self, learning_rate: float, gradient: np.ndarray) -> np.ndarray:
        if self.last is None:
            self.last = np.zeros(gradient.shape)
        self.last = self.gamma * self.last + (1 - self.gamma) * (gradient ** 2)
        return -learning_rate * gradient / np.sqrt(np.abs(self.last) + 0.00001)

    def __str__(self: Self) -> str:
        return f"{self.name}({self.gamma:.6g})"

class Adam(Optimization):
    name="Adam"

    def __init__(self: Self, beta_1: float = 0.9, beta_2: float = 0.999):
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.momentum_mean = 0
        self.momentum_var = 0
        self.timestep = 0

    @override
    def apply(self: Self, learning_rate: float, gradient: np.ndarray) -> np.ndarray:
        self.timestep += 1
        self.momentum_mean = self.beta_1 * self.momentum_mean + (1 - self.beta_1) * gradient
        self.momentum_var = self.beta_2 * self.momentum_var + (1 - self.beta_2) * gradient ** 2
        bias_corrected_mean = self.momentum_mean / (1 - self.beta_1 ** self.timestep)
        bias_corrected_var = self.momentum_var / (1 - self.beta_2 ** self.timestep)
        return -bias_corrected_mean * learning_rate / np.sqrt(np.abs(bias_corrected_var) + 0.00001)

    def __str__(self: Self) -> str:
        return f"{self.name}({self.beta_1:.6g},{self.beta_2:.6g})"
