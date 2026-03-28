from abc import ABC, abstractmethod
import numpy as np
## да простят меня питонисты за abstract_meta_factory_builders и прочее ооп зло
"""
интерфейс активации - forward для значения активации, backward- производная на обрабтном ходе сети
"""
class AbstractActivation(ABC):
    @abstractmethod
    def forward(self, input:np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def backward(self, output:np.ndarray) -> np.ndarray:
        raise NotImplementedError


