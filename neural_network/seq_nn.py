import numpy as np
from neural_network.nn_abstract import AbstractNeuralNetwork
from abstract_component import AbstractComponent

class SequentialNeuralNetwork(AbstractNeuralNetwork):
    """
    Полносвязная нейросеть последовательного типа.

    Состоит из списка компонентов, через которые данные проходят
    последовательно: линейное преобразование -> активация ->
    линейное преобразование и т.д

    Ожидает, что входы подаются батчем:
    inputs.shape == (batch_size, dim_in),
    а выборка и таргет согласованы по размерности:
    prediction.shape == target.shape
    """

    def __init__(self, components: list[AbstractComponent], loss):
        """
        Инициализирует сеть списком компонентов и функцией потерь.

        components — последовательность модулей сети, каждый из которых
        поддерживает forward, backward и update.
        loss — объект функции потерь, работающий с prediction и target
        одинаковой размерности.
        """
        self.components = components
        self.loss = loss

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Выполняет прямой проход по всем компонентам модели.

        Ожидаемая форма входа:
        inputs.shape == (batch_size, dim_in)

        Возвращает выход последнего компонента:
        output.shape == (batch_size, dim_out)
        """
        output = inputs
        for component in self.components:
            output = component.forward(output)
        return output

    def backward(self, prediction: np.ndarray, target: np.ndarray) -> None:
        """
        Выполняет обратный проход по всем компонентам сети

        Ожидается, что:
        prediction.shape == target.shape

        Начальный градиент берётся из функции потерь, затем последовательно
        передаётся в backward каждого компонента в обратном порядке
        Градиенты параметров сохраняются внутри обучаемых компонентов
        """
        gradient = self.loss.backward(prediction, target)
        for module in reversed(self.components):
            gradient = module.backward(gradient)

    def update(self, learning_rate: float) -> None:
        """
        Обновляет параметры всех компонентов модели.

        learning_rate — шаг градиентного спуска, общий для всех
        обучаемых компонентов
        """
        for component in self.components:
            component.update(learning_rate)

    def train_batch(
            self,
            inputs: np.ndarray,
            target: np.ndarray,
            learning_rate: float
    ) -> float:
        """
        Обучает сеть на одном батче.

        Ожидается, что:
        inputs.shape == (batch_size, dim_in)
        target.shape == prediction.shape

        Выполняет прямой проход, вычисление функции потерь,
        обратное распространение и обновление параметров

        Возвращает лосс для данного батча
        """
        prediction = self.forward(inputs)
        loss_value = self.loss.forward(prediction, target)
        self.backward(prediction, target)
        self.update(learning_rate)
        return loss_value

    def fit(
            self,
            inputs: np.ndarray,
            target: np.ndarray,
            epochs: int,
            learning_rate: float
    ) -> list[float]:
        """
        Обучает сеть на полном наборе данных заданное число эпох.

        Ожидается, что:
        inputs.shape == (batch_size, dim_in)
        target.shape == (batch_size, dim_out)

        В текущей реализации весь набор данных рассматривается как один батч.
        Каждая эпоха — один вызов train_batch.

        Возвращает список значений loss по эпохам.
        """
        history = []
        for _ in range(epochs):
            loss_value = self.train_batch(inputs, target, learning_rate)
            history.append(loss_value)
        return history

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Возвращает предсказание сети для входных данных.

        Ожидаемая форма входа:
        inputs.shape == (batch_size, dim_in)

        Возвращаемая форма:
        prediction.shape == (batch_size, dim_out)
        """
        return self.forward(inputs)