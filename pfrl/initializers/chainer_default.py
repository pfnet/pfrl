"""Initializes the weights and biases of a layer to chainer default.
"""
import torch
import torch.nn as nn

from pfrl.initializers.lecun_normal import init_lecun_normal


@torch.no_grad()
def init_chainer_default(layer):
    """Initializes the layer with the chainer default.
    weights with LeCunNormal(scale=1.0) and zeros as biases
    """
    assert isinstance(layer, nn.Module)

    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        init_lecun_normal(layer.weight)
        if layer.bias is not None:
            # layer may be initialized with bias=False
            nn.init.zeros_(layer.bias)
    return layer
