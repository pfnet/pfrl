import gym
import numpy as np


class CastObservation(gym.ObservationWrapper):
    """Cast observations to a given type.

    Args:
        env: Env to wrap.
        dtype: Data type object.

    Attributes:
        original_observation: Observation before casting.
    """

    def __init__(self, env, dtype):
        super().__init__(env)
        self.dtype = dtype

    def observation(self, observation):
        self.original_observation = observation
        return observation.astype(self.dtype, copy=False)


class CastObservationToFloat32(CastObservation):
    """Cast observations to float32, which is commonly used for NNs.

    Args:
        env: Env to wrap.

    Attributes:
        original_observation: Observation before casting.
    """

    def __init__(self, env):
        super().__init__(env, np.float32)
