"""Noisy Networks

See http://arxiv.org/abs/1706.10295
"""

import torch.nn as nn

from pfrl.nn.noisy_linear import FactorizedNoisyLinear


def to_factorized_noisy(module, *args, **kwargs):
    """Add noisiness to components of given module

    Currently this fn. only supports torch.nn.Linear (with and without bias)
    """

    def func_to_factorized_noisy(module):
        if isinstance(module, nn.Linear):
            return FactorizedNoisyLinear(module, *args, **kwargs)
        else:
            return module

    _map_modules(func_to_factorized_noisy, module)


def _map_modules(func, module):
    for name, child in module.named_children():
        new_child = func(child)
        if new_child is child:
            # It's not nn.Linear, so recurse
            _map_modules(func, child)
        else:
            module._modules[name] = new_child
