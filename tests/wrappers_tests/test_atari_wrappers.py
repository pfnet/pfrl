"""Currently this script tests `pfrl.wrappers.atari_wrappers.FrameStack`
only."""


from unittest import mock

import gym
import gym.spaces
import numpy as np
import pytest

from pfrl.wrappers.atari_wrappers import FrameStack, LazyFrames, ScaledFloatFrame


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("k", [2, 3])
def test_frame_stack(dtype, k):

    steps = 10

    # Mock env that returns atari-like frames
    def make_env(idx):
        env = mock.Mock()
        np_random = np.random.RandomState(idx)
        if dtype is np.uint8:

            def dtyped_rand():
                return np_random.randint(low=0, high=255, size=(1, 84, 84), dtype=dtype)

            low, high = 0, 255
        elif dtype is np.float32:

            def dtyped_rand():
                return np_random.rand(1, 84, 84).astype(dtype)

            low, high = -1.0, 3.14
        else:
            assert False
        env.reset.side_effect = [dtyped_rand() for _ in range(steps)]
        env.step.side_effect = [
            (
                dtyped_rand(),
                np_random.rand(),
                bool(np_random.randint(2)),
                {},
            )
            for _ in range(steps)
        ]
        env.action_space = gym.spaces.Discrete(2)
        env.observation_space = gym.spaces.Box(
            low=low, high=high, shape=(1, 84, 84), dtype=dtype
        )
        return env

    env = make_env(42)
    fs_env = FrameStack(make_env(42), k=k, channel_order="chw")

    # check action/observation space
    assert env.action_space == fs_env.action_space
    assert env.observation_space.dtype is fs_env.observation_space.dtype
    assert env.observation_space.low.item(0) == fs_env.observation_space.low.item(0)
    assert env.observation_space.high.item(0) == fs_env.observation_space.high.item(0)

    # check reset
    obs = env.reset()
    fs_obs = fs_env.reset()
    assert isinstance(fs_obs, LazyFrames)
    np.testing.assert_allclose(
        obs.take(indices=0, axis=fs_env.stack_axis),
        np.asarray(fs_obs).take(indices=0, axis=fs_env.stack_axis),
    )

    # check step
    for _ in range(steps - 1):
        action = env.action_space.sample()
        fs_action = fs_env.action_space.sample()
        obs, r, done, info = env.step(action)
        fs_obs, fs_r, fs_done, fs_info = fs_env.step(fs_action)
        assert isinstance(fs_obs, LazyFrames)
        np.testing.assert_allclose(
            obs.take(indices=0, axis=fs_env.stack_axis),
            np.asarray(fs_obs).take(indices=-1, axis=fs_env.stack_axis),
        )
        assert r == fs_r
        assert done == fs_done


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_scaled_float_frame(dtype):

    steps = 10

    # Mock env that returns atari-like frames
    def make_env(idx):
        env = mock.Mock()
        np_random = np.random.RandomState(idx)
        if dtype is np.uint8:

            def dtyped_rand():
                return np_random.randint(low=0, high=255, size=(1, 84, 84), dtype=dtype)

            low, high = 0, 255
        elif dtype is np.float32:

            def dtyped_rand():
                return np_random.rand(1, 84, 84).astype(dtype)

            low, high = -1.0, 3.14
        else:
            assert False
        env.reset.side_effect = [dtyped_rand() for _ in range(steps)]
        env.step.side_effect = [
            (
                dtyped_rand(),
                np_random.rand(),
                bool(np_random.randint(2)),
                {},
            )
            for _ in range(steps)
        ]
        env.action_space = gym.spaces.Discrete(2)
        env.observation_space = gym.spaces.Box(
            low=low, high=high, shape=(1, 84, 84), dtype=dtype
        )
        return env

    env = make_env(42)
    s_env = ScaledFloatFrame(make_env(42))

    # check observation space
    assert type(env.observation_space) is type(s_env.observation_space)  # NOQA
    assert s_env.observation_space.dtype is np.dtype(np.float32)
    assert s_env.observation_space.contains(s_env.observation_space.low)
    assert s_env.observation_space.contains(s_env.observation_space.high)

    # check reset
    obs = env.reset()
    s_obs = s_env.reset()
    np.testing.assert_allclose(np.array(obs) / s_env.scale, s_obs)

    # check step
    for _ in range(steps - 1):
        action = env.action_space.sample()
        s_action = s_env.action_space.sample()
        obs, r, done, info = env.step(action)
        s_obs, s_r, s_done, s_info = s_env.step(s_action)
        np.testing.assert_allclose(np.array(obs) / s_env.scale, s_obs)
        assert r == s_r
        assert done == s_done
