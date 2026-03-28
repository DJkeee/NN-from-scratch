import numpy as np
from .abstract_layer import AbstractLayer
from initialization.xavier_initializer import XavierInitializer
from initialization.zeros_initializer import ZerosInitializer


class DenseLayer(AbstractLayer):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        weight_initializer=XavierInitializer,
        bias_initializer=ZerosInitializer
    ):
        self.dim_in = dim_in
        self.dim_out = dim_out
#проверки чтоб инит был только при соблюдении контракта
        if isinstance(weight_initializer, type):
            weight_initializer = weight_initializer().initialize
        if isinstance(bias_initializer, type):
            bias_initializer = bias_initializer().initialize

        self.weights = np.asarray(weight_initializer((dim_in, dim_out)), dtype=float)
        self.bias = np.asarray(bias_initializer((1, dim_out)), dtype=float)

        self.input = None
        self.weights_gradient = None
        self.bias_gradient = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.input = inputs
        return inputs @ self.weights + self.bias

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        self.weights_gradient = self.input.T @ output_gradient
        self.bias_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        return output_gradient @ self.weights.T

    def update(self, learning_rate: float) -> None:
        self.weights -= learning_rate * self.weights_gradient
        self.bias -= learning_rate * self.bias_gradient