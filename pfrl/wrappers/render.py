import gym


class Render(gym.Wrapper):
    """Render env by calling its render method.

    Args:
        env (gym.Env): Env to wrap.
        **kwargs: Keyword arguments passed to the render method.
    """

    def __init__(self, env, **kwargs):
        super().__init__(env)
        self._kwargs = kwargs

    def reset(self, **kwargs):
        ret = self.env.reset(**kwargs)
        self.env.render(**self._kwargs)
        return ret

    def step(self, action):
        ret = self.env.step(action)
        self.env.render(**self._kwargs)
        return ret


# NEED TO REMOVE THIS LATER AND FIX THE ANTENV
class GymLikeEnvRender():
    """Render env by calling its render method.

    Args:
        env (gym.Env): Env to wrap.
        **kwargs: Keyword arguments passed to the render method.
    """

    def __init__(self, env, **kwargs):
        self.env = env
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.subgoal_dim = env.subgoal_dim

        self.action_space = env.action_space
        self.subgoal_space = env.subgoal_space

        self._kwargs = kwargs

    def reset(self, **kwargs):
        ret = self.env.reset(**kwargs)
        self.env.render(**self._kwargs)
        return ret

    def step(self, action):
        ret = self.env.step(action)
        self.env.render(**self._kwargs)
        return ret
