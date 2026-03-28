import numpy as np
from numpy import floating
from loss_functions.abstract_loss import AbstractLoss

class MSELoss(AbstractLoss):
    def forward(self, predict: np.ndarray, target: np.ndarray) -> floating:
        self.is_dims_equal(predict, predict)
        return np.mean((predict - target) ** 2)

    def backward(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        self.is_dims_equal(predict, target)
        return (2.0/predict.size) * (predict - target)

    def is_dims_equal(self, predict_vec: np.ndarray, target_vec: np.ndarray) -> bool:
        if target_vec.shape == predict_vec.shape:
            return True
        raise ValueError("проверь размерности векторов")