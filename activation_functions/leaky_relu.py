import abstract_activation
import numpy as np

class LeakyReLU(abstract_activation.AbstractActivation):
    def __init__(self, alpha: float = 0.01):
        if alpha < 0:
            raise ValueError("результат leakyrelu должен быть отрицательным - поменяйте при x < 0, поменяйте alpha")
        self.output = None
        self.alpha = alpha

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.output = np.where(inputs > 0, inputs, self.alpha * inputs)
        return self.output

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        return output_gradient * np.where(self.output > 0, 1.0, self.alpha)