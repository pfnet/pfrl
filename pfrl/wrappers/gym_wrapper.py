import gymnasium


class GymWrapper(gymnasium.Env):
    def __init__(self, gym_env):
        """A Gymnasium environment that wraps OpenAI gym environments."""
        super(GymWrapper, self).__init__()
        self.env = gym_env

    def reset(self, **kwargs):
        obs = self.env.reset()
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, False, info
