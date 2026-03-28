from abc import ABC, abstractmethod
import numpy as np


class AbstractComponent(ABC):
    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def update(self, learning_rate: float) -> None:
        pass