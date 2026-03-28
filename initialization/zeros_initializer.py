import numpy as np
from initialization.abstract_initializer import AbstractInitializer


class ZerosInitializer(AbstractInitializer):
    def initialize(self, shape: tuple[int, ...]) -> np.ndarray:
        return np.zeros(shape, dtype=float)