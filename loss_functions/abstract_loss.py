from abc import ABC, abstractmethod
import numpy as np

"""
Интерфейс для функций потерь

Задаёт общий контракт для вычисления значения функции потерь,
её градиента по предикту - на forward вычисление функции, на backward - градиент
так же является подтиптом компонента системы, то есть его можно использовать в components при создании сети
"""


class AbstractLoss(ABC):
    @abstractmethod
    def forward(self, predict_vec: np.ndarray, target_vec: np.ndarray):
        """
        Вычисляет значение функции потерь.

        Ожидается, что:
        predict_vec.shape == target_vec.shape

        Возвращает скалярную величину ошибки для пары
        предиикт / таргет
        """
        raise NotImplementedError

    @abstractmethod
    def backward(self, predict_vec: np.ndarray, target_vec: np.ndarray):
        """
        Вычисляет градиент функции потерь по предсказанию

        Ожидается, что:
        predict_vec.shape == target_vec.shape

        Возвращает массив той же формы, что и predict_vec,
        содержащий градиент функции потерь по каждому элементу предикта
        """
        raise NotImplementedError

    @abstractmethod
    def is_dims_equal(self, predict_vec: np.ndarray, target_vec: np.ndarray):
        """
        Проверяет совпадение размерностей предсказания и целевого значения

        Ожидается, что:
        predict_vec.shape == target_vec.shape

        Если размерности совпадают, проверка считается пройденной.
        Иначе выбрасывается исключение
        """
        if predict_vec.shape == predict_vec.shape:
            return True
        else:
            raise ValueError("проверь размерности векторов")