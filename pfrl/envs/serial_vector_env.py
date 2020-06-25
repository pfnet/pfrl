import numpy as np

import pfrl


class SerialVectorEnv(pfrl.env.VectorEnv):
    """VectorEnv where each env is run sequentially.

    The purpose of this VectorEnv is to help debugging. For speed, you should
    use MultiprocessVectorEnv if possible.

    Args:
        env_fns (list of gym.Env): List of gym.Env.
    """

    def __init__(self, envs):
        self.envs = envs
        self.last_obs = [None] * self.num_envs
        self.action_space = envs[0].action_space
        self.observation_space = envs[0].observation_space
        self.spec = envs[0].observation_space

    def step(self, actions):
        results = [env.step(a) for env, a in zip(self.envs, actions)]
        self.last_obs, rews, dones, infos = zip(*results)
        return self.last_obs, rews, dones, infos

    def reset(self, mask=None):
        if mask is None:
            mask = np.zeros(self.num_envs)
        obs = [
            env.reset() if not m else o
            for m, env, o in zip(mask, self.envs, self.last_obs)
        ]
        self.last_obs = obs
        return obs

    def seed(self, seeds):
        for env, seed in zip(self.envs, seeds):
            env.seed(seed)

    def close(self):
        for env in self.envs:
            env.close()

    @property
    def num_envs(self):
        return len(self.envs)
