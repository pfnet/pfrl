import gymnasium


class Render(gymnasium.Wrapper):
    """Render env by calling its render method.

    Args:
        env (gymnasium.Env): Env to wrap.
        **kwargs: Keyword arguments passed to the render method.
    """

    def __init__(self, env, **kwargs):
        super().__init__(env)
        self._kwargs = kwargs

    def reset(self, **kwargs):
        ret = self.env.reset(**kwargs)
        self.env.render(**self._kwargs)
        return ret, {}

    def step(self, action):
        ret = self.env.step(action)
        self.env.render(**self._kwargs)
        return ret
