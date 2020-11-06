import numpy as np
import torch
import torch.nn.functional as F

import pfrl


class Boltzmann(pfrl.explorer.Explorer):
    """Boltzmann exploration.

    Args:
        T (float): Temperature of Boltzmann distribution.
    """

    def __init__(self, T=1.0):
        self.T = T

    def select_action(self, t, greedy_action_func, action_value=None):
        assert action_value is not None
        assert isinstance(action_value, pfrl.action_value.DiscreteActionValue)
        n_actions = action_value.q_values.shape[1]
        with torch.no_grad():
            probs = (
                F.softmax(action_value.q_values / self.T, dim=-1).cpu().numpy().ravel()
            )
        return np.random.choice(np.arange(n_actions), p=probs)

    def __repr__(self):
        return "Boltzmann(T={})".format(self.T)
