from abc import ABC, abstractmethod
import numpy as np


class AbstractInitializer(ABC):
    @abstractmethod
    def initialize(self, shape: tuple[int, ...]) -> np.ndarray:
        raise NotImplementedError