from abc import ABCMeta, abstractmethod


class StateQFunction(object, metaclass=ABCMeta):
    """Abstract Q-function with state input."""

    @abstractmethod
    def __call__(self, x):
        """Evaluates Q-function

        Args:
            x (ndarray): state input

        Returns:
            An instance of ActionValue that allows to calculate the Q-values
            for state x and every possible action
        """
        raise NotImplementedError()


class StateActionQFunction(object, metaclass=ABCMeta):
    """Abstract Q-function with state and action input."""

    @abstractmethod
    def __call__(self, x, a):
        """Evaluates Q-function

        Args:
            x (ndarray): state input
            a (ndarray): action input

        Returns:
            Q-value for state x and action a
        """
        raise NotImplementedError()
