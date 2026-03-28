import numpy as np
from neural_network.nn_abstract import AbstractNeuralNetwork


class SequentialNeuralNetwork(AbstractNeuralNetwork):
    def __init__(self, modules: list, loss):
        self.modules = modules
        self.loss = loss

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        output = inputs
        for module in self.modules:
            output = module.forward(output)
        return output

    def backward(self, prediction: np.ndarray, target: np.ndarray) -> None:
        gradient = self.loss.backward(prediction, target)
        for module in reversed(self.modules):
            gradient = module.backward(gradient)

    def update(self, learning_rate: float) -> None:
        for module in self.modules:
            if hasattr(module, "update"):
                module.update(learning_rate)

    def train_batch(
        self,
        inputs: np.ndarray,
        target: np.ndarray,
        learning_rate: float
    ) -> float:
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
        history = []
        for _ in range(epochs):
            loss_value = self.train_batch(inputs, target, learning_rate)
            history.append(loss_value)
        return history

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return self.forward(inputs)