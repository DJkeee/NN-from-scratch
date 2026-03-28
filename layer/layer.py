import numpy as np
from abstract_layer import AbstractLayer
from initialization.xavier_initializer import XavierInitializer
from initialization.zeros_initializer import ZerosInitializer
"""
Полносвязный линейный слой.

Выполняет линейное преобразование:
output = inputs * weights + bias (векторно)

Хранит матрицу весов, вектор смещений и градиенты этих параметров,
вычисленные на бэкпропе
"""


class DenseLayer(AbstractLayer):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        weight_initializer=XavierInitializer,
        bias_initializer=ZerosInitializer
    ):
        """
        Инициализирует полносвязный слой.

        dim_in — размер входного признакового пространства.
        dim_out — размер выходного признакового пространства.

        weight_initializer — способ инициализации матрицы весов
        формы (dim_in, dim_out).
        bias_initializer — способ инициализации смещения
        формы (1, dim_out).

        После создания слой содержит:
        weights.shape == (dim_in, dim_out)
        bias.shape == (1, dim_out)
        """
        self.dim_in = dim_in
        self.dim_out = dim_out

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
        """
        Выполняет прямой проход через слой.

        Ожидаемая форма входа:
        inputs.shape == (batch_size, dim_in)

        Возвращает:
        output.shape == (batch_size, dim_out)

        На прямом проходе вход сохраняется в self.input,
        так как он нужен для вычисления градиента весов
        на обратном проходе.
        """
        self.input = inputs
        return inputs @ self.weights + self.bias

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        """
        Выполняет обратный проход через слой.

        Принимает градиент функции потерь по выходу слоя:
        output_gradient.shape == (batch_size, dim_out)

        Вычисляет и сохраняет:
        weights_gradient.shape == (dim_in, dim_out)
        bias_gradient.shape == (1, dim_out)

        Возвращает градиент функции потерь по входу слоя:
        input_gradient.shape == (batch_size, dim_in)
        """
        self.weights_gradient = self.input.T @ output_gradient
        self.bias_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        return output_gradient @ self.weights.T

    def update(self, learning_rate: float) -> None:
        """
        Обновляет параметры слоя по ранее вычисленным градиентам.

        learning_rate — шаг градиентного спуска.

        После вызова метода изменяются:
        weights
        bias
        """
        self.weights -= learning_rate * self.weights_gradient
        self.bias -= learning_rate * self.bias_gradient