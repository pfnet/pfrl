from collections import deque

import numpy as np
from gym import spaces

from pfrl.env import VectorEnv
from pfrl.wrappers.atari_wrappers import LazyFrames


class VectorEnvWrapper(VectorEnv):
    """VectorEnv analog to gym.Wrapper."""

    def __init__(self, env):
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(
                "attempted to get missing private attribute '{}'".format(name)
            )
        return getattr(self.env, name)

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def render(self, mode="human", **kwargs):
        return self.env.render(mode, **kwargs)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.env.compute_reward(achieved_goal, desired_goal, info)

    def __str__(self):
        return "<{}{}>".format(type(self).__name__, self.env)

    def __repr__(self):
        return str(self)

    @property
    def unwrapped(self):
        return self.env.unwrapped


class VectorFrameStack(VectorEnvWrapper):
    """VectorEnv analog to pfrl.wrappers.atari_wrappers.FrameStack.

    The original `pfrl.wrappers.atari_wrappers.FrameStack` does not work
    properly with `pfrl.envs.MultiprocessVectorEnv` because LazyFrames
    becomes not lazy when passed between processes, unnecessarily increasing
    memory usage. To avoid the issue, use this wrapper instead of `FrameStack`
    so that LazyFrames are not passed between processes.

    Args:
        env (VectorEnv): Env to wrap.
        k (int): How many frames to stack.
        stack_axis (int): Axis along which frames are concatenated.
    """

    def __init__(self, env, k, stack_axis=0):
        """Stack k last frames."""
        VectorEnvWrapper.__init__(self, env)
        self.k = k
        self.stack_axis = stack_axis
        self.frames = [deque([], maxlen=k) for _ in range(env.num_envs)]
        orig_obs_space = env.observation_space
        assert isinstance(orig_obs_space, spaces.Box)
        low = np.repeat(orig_obs_space.low, k, axis=self.stack_axis)
        high = np.repeat(orig_obs_space.high, k, axis=self.stack_axis)
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=orig_obs_space.dtype
        )

    def reset(self, mask=None):
        batch_ob = self.env.reset(mask=mask)
        if mask is None:
            mask = np.zeros(self.env.num_envs)
        for m, frames, ob in zip(mask, self.frames, batch_ob):
            if not m:
                for _ in range(self.k):
                    frames.append(ob)
        return self._get_ob()

    def step(self, action):
        batch_ob, reward, done, info = self.env.step(action)
        for frames, ob in zip(self.frames, batch_ob):
            frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.env.num_envs
        assert len(self.frames[0]) == self.k
        return [
            LazyFrames(list(frames), stack_axis=self.stack_axis)
            for frames in self.frames
        ]
