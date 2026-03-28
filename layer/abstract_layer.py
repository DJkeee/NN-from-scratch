from abc import ABC, abstractmethod
import numpy as np

class AbstractLayer(ABC):
    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def update(self, learning_rate: float) -> None:
        raise NotImplementedError