from abc import ABCMeta, abstractmethod

import numpy as np


class StepHook(object, metaclass=ABCMeta):
    """Hook function that will be called in training.

    This class is for clarifying the interface required for Hook functions.
    You don't need to inherit this class to define your own hooks. Any callable
    that accepts (env, agent, step) as arguments can be used as a hook.
    """

    @abstractmethod
    def __call__(self, env, agent, step):
        """Call the hook.

        Args:
            env: Environment.
            agent: Agent.
            step: Current timestep.
        """
        raise NotImplementedError


class LinearInterpolationHook(StepHook):
    """Hook that will set a linearly interpolated value.

    You can use this hook to decay the learning rate by using a setter function
    as follows:

    .. code-block:: python

        def lr_setter(env, agent, value):
            agent.optimizer.lr = value

        hook = LinearInterpolationHook(10 ** 6, 1e-3, 0, lr_setter)


    Args:
        total_steps (int): Number of total steps.
        start_value (float): Start value.
        stop_value (float): Stop value.
        setter (callable): (env, agent, value) -> None
    """

    def __init__(self, total_steps, start_value, stop_value, setter):
        self.total_steps = total_steps
        self.start_value = start_value
        self.stop_value = stop_value
        self.setter = setter

    def __call__(self, env, agent, step):
        value = np.interp(
            step, [1, self.total_steps], [self.start_value, self.stop_value]
        )
        self.setter(env, agent, value)
