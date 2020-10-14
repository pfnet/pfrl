import signal
import warnings
from multiprocessing import Pipe, Process

import numpy as np
from torch.distributions.utils import lazy_property

import pfrl


def worker(remote, env_fn):
    # Ignore CTRL+C in the worker process
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    env = env_fn()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                ob, reward, done, info = env.step(data)
                remote.send((ob, reward, done, info))
            elif cmd == "reset":
                ob = env.reset()
                remote.send(ob)
            elif cmd == "close":
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.action_space, env.observation_space))
            elif cmd == "spec":
                remote.send(env.spec)
            elif cmd == "seed":
                remote.send(env.seed(data))
            else:
                raise NotImplementedError
    finally:
        env.close()


class MultiprocessVectorEnv(pfrl.env.VectorEnv):
    """VectorEnv where each env is run in its own subprocess.

    Args:
        env_fns (list of callable): List of callables, each of which
            returns gym.Env that is run in its own subprocess.
    """

    def __init__(self, env_fns):
        if np.__version__ == "1.16.0":
            warnings.warn(
                """
NumPy 1.16.0 can cause severe memory leak in pfrl.envs.MultiprocessVectorEnv.
We recommend using other versions of NumPy.
See https://github.com/numpy/numpy/issues/12793 for details.
"""
            )  # NOQA

        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [
            Process(target=worker, args=(work_remote, env_fn))
            for (work_remote, env_fn) in zip(self.work_remotes, env_fns)
        ]
        for p in self.ps:
            p.start()
        self.last_obs = [None] * self.num_envs
        self.remotes[0].send(("get_spaces", None))
        self.action_space, self.observation_space = self.remotes[0].recv()
        self.closed = False

    def __del__(self):
        if not self.closed:
            self.close()

    @lazy_property
    def spec(self):
        self._assert_not_closed()
        self.remotes[0].send(("spec", None))
        spec = self.remotes[0].recv()
        return spec

    def step(self, actions):
        self._assert_not_closed()
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        results = [remote.recv() for remote in self.remotes]
        self.last_obs, rews, dones, infos = zip(*results)
        return self.last_obs, rews, dones, infos

    def reset(self, mask=None):
        self._assert_not_closed()
        if mask is None:
            mask = np.zeros(self.num_envs)
        for m, remote in zip(mask, self.remotes):
            if not m:
                remote.send(("reset", None))

        obs = [
            remote.recv() if not m else o
            for m, remote, o in zip(mask, self.remotes, self.last_obs)
        ]
        self.last_obs = obs
        return obs

    def close(self):
        self._assert_not_closed()
        self.closed = True
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()

    def seed(self, seeds=None):
        self._assert_not_closed()
        if seeds is not None:
            if isinstance(seeds, int):
                seeds = [seeds] * self.num_envs
            elif isinstance(seeds, list):
                if len(seeds) != self.num_envs:
                    raise ValueError(
                        "length of seeds must be same as num_envs {}".format(
                            self.num_envs
                        )
                    )
            else:
                raise TypeError(
                    "Type of Seeds {} is not supported.".format(type(seeds))
                )
        else:
            seeds = [None] * self.num_envs

        for remote, seed in zip(self.remotes, seeds):
            remote.send(("seed", seed))
        results = [remote.recv() for remote in self.remotes]
        return results

    @property
    def num_envs(self):
        return len(self.remotes)

    def _assert_not_closed(self):
        assert not self.closed, "This env is already closed"
