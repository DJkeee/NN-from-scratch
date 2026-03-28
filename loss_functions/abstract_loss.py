from abc import ABC, abstractmethod
import numpy as np
"""
интефейс лоссов - ничего интересного
"""

class AbstractLoss(ABC):
    @abstractmethod
    def forward(self, predict_vec:np.ndarray, target_vec:np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def backward(self, predict_vec:np.ndarray, target_vec:np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def is_dims_equal(self, predict_vec:np.ndarray, target_vec:np.ndarray):
        if predict_vec.shape == predict_vec.shape:
            return True
        else:
            raise ValueError("проверь размерности векторов")

