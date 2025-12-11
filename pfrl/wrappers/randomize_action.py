import gymnasium
import numpy as np


class RandomizeAction(gymnasium.ActionWrapper):
    """Apply a random action instead of the one sent by the agent.

    This wrapper can be used to make a stochastic env. The common use is
    for evaluation in Atari environments, where actions are replaced with
    random ones with a low probability.

    Only gymnasium.spaces.Discrete is supported as an action space.

    For exploration during training, use explorers like
    pfrl.explorers.ConstantEpsilonGreedy instead of this wrapper.

    Args:
        env (gymnasium.Env): Env to wrap.
        random_fraction (float): Fraction of actions that will be replaced
            with a random action. It must be in [0, 1].
    """

    def __init__(self, env, random_fraction):
        super().__init__(env)
        assert 0 <= random_fraction <= 1
        assert isinstance(
            env.action_space, gymnasium.spaces.Discrete
        ), "RandomizeAction supports only gymnasium.spaces.Discrete as an action space"
        self._random_fraction = random_fraction
        self._rng = np.random.RandomState()

    def action(self, action):
        if self._rng.rand() < self._random_fraction:
            return self._rng.randint(self.env.action_space.n)
        else:
            return action

    def reset(self, **kwargs):
        if 'seed' in kwargs:
            self._rng = np.random.RandomState(kwargs['seed'])
        return self.env.reset(**kwargs)

