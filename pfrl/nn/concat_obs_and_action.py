import torch

from pfrl.nn.lmbda import Lambda


def concat_obs_and_action(obs_and_action):
    """Concat observation and action to feed the critic."""
    assert len(obs_and_action) == 2
    return torch.cat(obs_and_action, dim=-1)


class ConcatObsAndAction(Lambda):
    def __init__(self):
        return super().__init__(concat_obs_and_action)
