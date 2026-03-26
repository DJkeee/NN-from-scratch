import abstract_activation
import numpy as np


class Tanh(abstract_activation.AbstractActivation):
    def __init__(self):
        self.output = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.output = np.tanh(inputs)
        return self.output

# производная tanh(x) = 1 - tanh(x)**2 - удобно
    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        return output_gradient * (1.0 - self.output ** 2)