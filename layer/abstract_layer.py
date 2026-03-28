from abc import ABC, abstractmethod
import numpy as np

"""
интерфейс слоёв нейронной сети.

Определяет единый интерфейс для прямого прохода (вычисление выхода),
обратного прохода (вычисление градиентов) и обновления обучаемых параметров.
"""


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