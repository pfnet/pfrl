import gym


class ScaleReward(gym.RewardWrapper):
    """Scale reward by a scale factor.

    Args:
        env: Env to wrap.
        scale (float): Scale factor.

    Attributes:
        scale: Scale factor.
        original_reward: Reward before casting.
    """

    def __init__(self, env, scale):
        super().__init__(env)
        self.scale = scale
        self.original_reward = None

    def reward(self, reward):
        self.original_reward = reward
        return self.scale * reward
