import gym


class ContinuingTimeLimit(gym.Wrapper):
    """TimeLimit wrapper for continuing environments.

    This is similar gym.wrappers.TimeLimit, which sets a time limit for
    each episode, except that done=False is returned and that
    info['needs_reset'] is set to True when past the limit.

    Code that calls env.step is responsible for checking the info dict, the
    fourth returned value, and resetting the env if it has the 'needs_reset'
    key and its value is True.

    Args:
        env (gym.Env): Env to wrap.
        max_episode_steps (int): Maximum number of timesteps during an episode,
            after which the env needs a reset.
    """

    def __init__(self, env, max_episode_steps):
        super(ContinuingTimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps

        self._elapsed_steps = None

    def step(self, action):
        assert (
            self._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1

        if self._max_episode_steps <= self._elapsed_steps:
            info["needs_reset"] = True

        return observation, reward, done, info

    def reset(self):
        self._elapsed_steps = 0
        return self.env.reset()
