import gym
import gym.spaces
import numpy as np
import pytest

import pfrl


class ActionRecordingEnv(gym.Env):

    observation_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
    action_space = gym.spaces.Discrete(3)

    def __init__(self):
        self.past_actions = []

    def reset(self):
        return self.observation_space.sample()

    def step(self, action):
        self.past_actions.append(action)
        return self.observation_space.sample(), 0, False, {}


@pytest.mark.parametrize("random_fraction", [0, 0.3, 0.6, 1])
def test_action_ratio(random_fraction):
    env = ActionRecordingEnv()
    env = pfrl.wrappers.RandomizeAction(env, random_fraction=random_fraction)
    env.reset()
    n = 1000
    delta = 0.05
    for _ in range(n):
        # Always send action 0
        env.step(0)
    # Ratio of selected actions should be:
    #   0: (1 - random_fraction) + random_fraction/3
    #   1: random_fraction/3
    #   2: random_fraction/3
    np.testing.assert_allclose(
        env.env.past_actions.count(0) / n,
        (1 - random_fraction) + random_fraction / 3,
        atol=delta,
    )
    np.testing.assert_allclose(
        env.env.past_actions.count(1) / n, random_fraction / 3, atol=delta
    )
    np.testing.assert_allclose(
        env.env.past_actions.count(2) / n, random_fraction / 3, atol=delta
    )


@pytest.mark.parametrize("random_fraction", [0, 0.3, 0.6, 1])
def test_seed(random_fraction):
    def get_actions(seed):
        env = ActionRecordingEnv()
        env = pfrl.wrappers.RandomizeAction(env, random_fraction=random_fraction)
        env.seed(seed)
        for _ in range(1000):
            # Always send action 0
            env.step(0)
        return env.env.past_actions

    a_seed0 = get_actions(0)
    a_seed1 = get_actions(1)
    b_seed0 = get_actions(0)
    b_seed1 = get_actions(1)

    assert a_seed0 == b_seed0
    assert a_seed1 == b_seed1
    if random_fraction > 0:
        assert a_seed0 != a_seed1
