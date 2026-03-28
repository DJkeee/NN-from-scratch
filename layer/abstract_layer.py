from abc import ABC, abstractmethod
import numpy as np
from abstract_component import AbstractComponent

"""
Интерфейс слоёв нейронной сети

Определяет общий контракт для слоев нейронной сети - является подтипом компонента
Слой должен поддерживать прямой проход, обратный проход и обновление
параметров по вычисленным градиентам

так же является подтиптом компонента системы, то есть его можно использовать в components при создании сети
"""


class AbstractLayer(AbstractComponent):
    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Выполняет прямой проход через слой.

        Ожидаемая форма входа:
        inputs.shape == (batch_size, dim_in)

        Возвращает выход слоя:
        output.shape == (batch_size, dim_out)
        """
        raise NotImplementedError

    @abstractmethod
    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        """
        Выполняет обратный проход через слой.

        Принимает градиент функции потерь по выходу слоя:
        output_gradient.shape == (batch_size, dim_out)

        Возвращает градиент функции потерь по входу слоя:
        input_gradient.shape == (batch_size, dim_in)
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, learning_rate: float) -> None:
        """
        Обновляет обучаемые параметры слоя.

        learning_rate — шаг градиентного спуска, используемый
        для изменения параметров слоя на основе ранее вычисленных градиентов.
        """
        raise NotImplementedError