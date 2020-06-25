import gym
import numpy as np
import pytest

import pfrl


@pytest.mark.parametrize("env_id", ["CartPole-v1", "Pendulum-v0"])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_cast_observation(env_id, dtype):
    env = pfrl.wrappers.CastObservation(gym.make(env_id), dtype=dtype)
    rtol = 1e-3 if dtype == np.float16 else 1e-7

    obs = env.reset()
    assert env.original_observation.dtype == np.float64
    assert obs.dtype == dtype
    np.testing.assert_allclose(env.original_observation, obs, rtol=rtol)

    obs, r, done, info = env.step(env.action_space.sample())

    assert env.original_observation.dtype == np.float64
    assert obs.dtype == dtype
    np.testing.assert_allclose(env.original_observation, obs, rtol=rtol)


@pytest.mark.parametrize("env_id", ["CartPole-v1", "Pendulum-v0"])
def test_cast_observation_to_float32(env_id):
    env = pfrl.wrappers.CastObservationToFloat32(gym.make(env_id))

    obs = env.reset()
    assert env.original_observation.dtype == np.float64
    assert obs.dtype == np.float32
    np.testing.assert_allclose(env.original_observation, obs)

    obs, r, done, info = env.step(env.action_space.sample())
    assert env.original_observation.dtype == np.float64
    assert obs.dtype == np.float32
    np.testing.assert_allclose(env.original_observation, obs)
