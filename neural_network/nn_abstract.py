from abc import ABC, abstractmethod
import numpy as np


class AbstractNeuralNetwork(ABC):
    """
    интерфейс для полносвязной нейронной сети.

    Определяет основной интерфейс
    прямой проход для получения предсказаний, обратный проход для вычисления градиентов,
    обновление параметров, обучение на батче и на всей выборке через фит, а также предикт.
    """

    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Выполняет прямой проход по всем слоям сети и возвращает предикт
        """

    @abstractmethod
    def backward(self, prediction: np.ndarray, target: np.ndarray) -> None:
        """
        Вычисляет градиенты всех параметров сети через бэкпроп
        """

    @abstractmethod
    def update(self, learning_rate: float) -> None:
        """
        Обновляет все обучаемые параметры сети с учетом скорости обучения в бэкпропе
        """

    @abstractmethod
    def train_batch(
        self,
        inputs: np.ndarray,
        target: np.ndarray,
        learning_rate: float
    ) -> float:
        """
        Обучает сеть на одном батче
        """

    @abstractmethod
    def fit(
        self,
        inputs: np.ndarray,
        target: np.ndarray,
        epochs: int,
        learning_rate: float
    ) -> list[float]:
        """
        Обучает сеть на полном наборе данных в течение заданного числа эпох.
        Каждая эпоха может использовать батчи. Возвращает список
        значений функии потерь по каждой эпохе.
        """

    @abstractmethod
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Возвращает предикт сети
        """