import numpy as np

from pfrl import explorer


class AdditiveGaussian(explorer.Explorer):
    """Additive Gaussian noise to actions.

    Each action must be numpy.ndarray.

    Args:
        scale (float or array_like of floats): Scale parameter.
        low (float, array_like of floats, or None): Lower bound of action
            space used to clip an action after adding a noise. If set to None,
            clipping is not performed on lower edge.
        high (float, array_like of floats, or None): Higher bound of action
            space used to clip an action after adding a noise. If set to None,
            clipping is not performed on upper edge.
    """

    def __init__(self, scale, low=None, high=None):
        self.scale = scale
        self.low = low
        self.high = high

    def select_action(self, t, greedy_action_func, action_value=None):
        a = greedy_action_func()
        noise = np.random.normal(scale=self.scale, size=a.shape).astype(np.float32)
        if self.low is not None or self.high is not None:
            return np.clip(a + noise, self.low, self.high)
        else:
            return a + noise

    def __repr__(self):
        return "AdditiveGaussian(scale={}, low={}, high={})".format(
            self.scale, self.low, self.high
        )
