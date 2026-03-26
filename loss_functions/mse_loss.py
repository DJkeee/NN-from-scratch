import numpy as np
import abstract_loss

class MSELoss(abstract_loss.AbstractLoss):
    def forward(self, predict: np.ndarray, target: np.ndarray) -> np.float64:
        self.is_dims_equal(predict, predict)
        return np.mean((predict - target) ** 2)

    def backward(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        self.is_dims_equal(predict, target)
        return (2.0/predict.size) * (predict - target)

    def is_dims_equal(self, predict_vec: np.ndarray, target_vec: np.ndarray) -> bool:
        if target_vec.shape == predict_vec.shape:
            return True
        raise ValueError("проверь размерности векторов")