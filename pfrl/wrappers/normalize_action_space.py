import gym
import gym.spaces
import numpy as np


class NormalizeActionSpace(gym.ActionWrapper):
    """Normalize a Box action space to [-1, 1]^n."""

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Box)
        self.action_space = gym.spaces.Box(
            low=-np.ones_like(env.action_space.low),
            high=np.ones_like(env.action_space.low),
        )

    def action(self, action):
        # action is in [-1, 1]
        action = action.copy()

        # -> [0, 2]
        action += 1

        # -> [0, orig_high - orig_low]
        action *= (self.env.action_space.high - self.env.action_space.low) / 2

        # -> [orig_low, orig_high]
        return action + self.env.action_space.low
