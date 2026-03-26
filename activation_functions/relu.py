import abstract_activation
import numpy as np
class Relu(abstract_activation.AbstractActivation):
    def __init__(self):
        self.output = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.output = np.maximum(inputs, 0)
        return self.output

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        return output_gradient * (self.output > 0)
