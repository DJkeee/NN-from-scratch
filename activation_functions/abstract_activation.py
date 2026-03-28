from abc import abstractmethod
import numpy as np
from abstract_component import AbstractComponent

## да простят меня питонисты за abstract_meta_factory_builders и прочее ооп зло
"""
Интерфейс функций активации.

Определяет общий контракт для нелинейных компонентов сети ака функций активации.
forward отвечает за возвращения значения по входу,
backward - для значения производной, без нее градиента не будет(
"""

class AbstractActivation(AbstractComponent):
    @abstractmethod
    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Выполняет прямой проход через функцию активации.
        Возвращает результат применения активации
        требование к размерности:
        output.shape == input.shape
        """
        raise NotImplementedError

    @abstractmethod
    def backward(self, output: np.ndarray) -> np.ndarray:
        """
        Выполняет обратный проход через функцию активации.

        Принимает градиент функции потерь по выходу активации:
        output.shape == (batch_size, dim_out)

        Возвращает градиент функции потерь по входу
        той же размерности.
        """
        raise NotImplementedError

