import numpy as np
from initialization.abstract_initializer import AbstractInitializer

class XavierInitializer(AbstractInitializer):
    def initialize(self, shape: tuple[int, ...]) -> np.ndarray:
        dim_in, dim_out = shape
        return np.random.uniform(-np.sqrt(6.0 / (dim_in + dim_out)), np.sqrt(6.0 / (dim_in + dim_out)), size=shape)