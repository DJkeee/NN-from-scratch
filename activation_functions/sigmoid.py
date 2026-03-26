import abstract_activation
import numpy as np


class Sigmoid(abstract_activation.AbstractActivation):
    def __init__(self):
        self.output = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.output = 1.0 / (1.0 + np.exp(-inputs))
        return self.output

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        return output_gradient * self.output * (1.0 - self.output)