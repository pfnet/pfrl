class NormalizedRewardFilter(object):
    def __init__(self, tau=1e-3, scale=1, eps=1e-1):
        self.tau = tau
        self.scale = scale
        self.average_reward = 0
        self.average_reward_squared = 0
        self.eps = eps

    def __call__(self, reward):
        self.average_reward *= 1 - self.tau
        self.average_reward += self.tau * reward
        self.average_reward_squared *= 1 - self.tau
        self.average_reward_squared += self.tau * reward ** 2
        var = self.average_reward_squared - self.average_reward ** 2
        stdev = min(var, self.eps) ** 0.5
        return self.scale * (reward - self.average_reward) / stdev


class AverageRewardFilter(object):
    def __init__(self, tau=1e-3):
        self.tau = tau
        self.average_reward = 0

    def __call__(self, reward):
        self.average_reward *= 1 - self.tau
        self.average_reward += self.tau * reward
        return reward - self.average_reward
