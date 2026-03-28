from abc import ABC, abstractmethod
import numpy as np


class AbstractNeuralNetwork(ABC):
    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def backward(self, prediction: np.ndarray, target: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def update(self, learning_rate: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def train_batch(
        self,
        inputs: np.ndarray,
        target: np.ndarray,
        learning_rate: float
    ) -> float:
        raise NotImplementedError

    @abstractmethod
    def fit(
        self,
        inputs: np.ndarray,
        target: np.ndarray,
        epochs: int,
        learning_rate: float
    ) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError