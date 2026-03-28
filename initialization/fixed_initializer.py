import numpy as np
from initialization.abstract_initializer import AbstractInitializer

"""
Инициализатор для задания весов и биаса хардкодом

Используется в случаях, когда начальные значения параметров слоя
нужно задать явно,

Пример использования:

weights = np.array([
    [0.5, 0.2, 0.4],
    [0.5, 0.2, 0.4]
])

bias = np.array([[0.06, 0.06, 0.06]])

layer = DenseLayer(
    dim_in=2,
    dim_out=3,
    weight_initializer=HardCodeInitializer(weights),
    bias_initializer=HardCodeInitializer(bias)
)
"""


class HardCodeInitializer(AbstractInitializer):
    """
    Инициализатор, возвращающий заранее заданный массив значений.

    Хранит фиксированные значения параметров и возвращает их при вызове
    initialize, если их размерность совпадает с ожидаемой.
    """

    def __init__(self, values: np.ndarray):
        """
        Сохраняет массив фиксированных значений инициализации.

        values — заранее заданные веса или смещения,
        которые будут использованы при создании параметров слоя.
        """
        self.values = np.asarray(values, dtype=float)

    def initialize(self, shape: tuple[int, ...]) -> np.ndarray:
        """
        Возвращает заранее заданные значения параметров.

        Ожидается, что:
        self.values.shape == shape

        Если размерность совпадает, возвращается копия массива
        фиксированных значений. Иначе выбрасывается исключение.
        важно что массив копируется - если решите использовать один массив весов для n слоев,
        то при обучении не будет меняться переданный массив. не убирать
        """
        if self.values.shape != shape:
            raise ValueError(f"ожидалась размерность {shape}, получили {self.values.shape}")
        return self.values.copy()