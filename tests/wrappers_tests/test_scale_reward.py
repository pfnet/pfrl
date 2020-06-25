import gym
import numpy as np
import pytest

import pfrl


@pytest.mark.parametrize("env_id", ["CartPole-v1", "MountainCar-v0"])
@pytest.mark.parametrize("scale", [1.0, 0.1])
def test_scale_reward(env_id, scale):
    env = pfrl.wrappers.ScaleReward(gym.make(env_id), scale=scale)
    assert env.original_reward is None
    np.testing.assert_allclose(env.scale, scale)

    _ = env.reset()
    _, r, _, _ = env.step(env.action_space.sample())

    if env_id == "CartPole-v1":
        # Original reward must be 1
        np.testing.assert_allclose(env.original_reward, 1)
        np.testing.assert_allclose(r, scale)
    elif env_id == "MountainCar-v0":
        # Original reward must be -1
        np.testing.assert_allclose(env.original_reward, -1)
        np.testing.assert_allclose(r, -scale)
    else:
        assert False
