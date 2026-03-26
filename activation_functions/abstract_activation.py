from abc import ABC, abstractmethod
import numpy as np
## да простят меня питонисты за abstract_meta_factory_builders и прочее ооп зло
class AbstractActivation(ABC):
    @abstractmethod
    def forward(self, input:np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def backward(self, output:np.ndarray) -> np.ndarray:
        raise NotImplementedError


