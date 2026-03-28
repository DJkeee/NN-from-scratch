import numpy as np
from neural_network.nn_abstract import AbstractNeuralNetwork


class SequentialNeuralNetwork(AbstractNeuralNetwork):
    """
    Полносвязная нейронка sequence типа

    Состоит из списка модулей(список слоев), через которые
    данные проходят как линейное преобразование->активация->лин преобраз и тд. Управляет прямым проходом, обратным распространением,
    обновлением параметров,позволяет обучатьна батче и полной выборке
    """

    def __init__(self, modules: list, loss):
        """
        Инициализирует сеть заданными модулями(модули - слои активации и линейные слои) и функцией потерь
        """
        self.modules = modules
        self.loss = loss

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Выполняет прямой проход в модели.
        """
        output = inputs
        for module in self.modules:
            output = module.forward(output)
        return output

    def backward(self, prediction: np.ndarray, target: np.ndarray) -> None:
        """
        Выполняет обратный проход: вычисляет градиенты всех параметров сети.
        Градиенты накапливаются внутри каждого модуля.
        """
        gradient = self.loss.backward(prediction, target)
        for module in reversed(self.modules):
            gradient = module.backward(gradient)

    def update(self, learning_rate: float) -> None:
        """
        Обновляет все обучаемые параметры частей модели
        """
        for module in self.modules:
            if hasattr(module, "update"):
                module.update(learning_rate)

    def train_batch(
        self,
        inputs: np.ndarray,
        target: np.ndarray,
        learning_rate: float
    ) -> float:
        """
        Обучает сеть на одном мини-батче.

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
        Обучает сеть на полном наборе данных в течение заданного числа эпох.

        Каждая эпоха состоит из одного вызова train_batch (т.е. весь набор данных
        считается одним батчем). Возвращает историю значений функции потерь по эпохам.
        """
        history = []
        for _ in range(epochs):
            loss_value = self.train_batch(inputs, target, learning_rate)
            history.append(loss_value)
        return history

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Возвращает предсказание сети для входных данных.
        """
        return self.forward(inputs)