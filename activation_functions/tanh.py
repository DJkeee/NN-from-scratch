import numpy as np
from activation_functions.abstract_activation import AbstractActivation

class Tanh(AbstractActivation):
    def __init__(self):
        self.output = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.output = np.tanh(inputs)
        return self.output

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        return output_gradient * (1.0 - self.output ** 2)