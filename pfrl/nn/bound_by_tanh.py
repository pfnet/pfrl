from pfrl.functions.bound_by_tanh import bound_by_tanh
from pfrl.nn.lmbda import Lambda


class BoundByTanh(Lambda):
    """Bound a given value into [low, high] by tanh.

    Args:
        low (numpy.ndarray): lower bound
        high (numpy.ndarray): upper bound
    """

    def __init__(self, low, high):
        super().__init__(lambda x: bound_by_tanh(x, low, high))
