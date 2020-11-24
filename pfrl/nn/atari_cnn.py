import torch
import torch.nn as nn
import torch.nn.functional as F

from pfrl.initializers import init_chainer_default


def constant_bias_initializer(bias=0.0):
    @torch.no_grad()
    def init_bias(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            m.bias.fill_(bias)

    return init_bias


class LargeAtariCNN(nn.Module):
    """Large CNN module proposed for DQN in Nature, 2015.

    See: https://www.nature.com/articles/nature14236
    """

    def __init__(
        self, n_input_channels=4, n_output_channels=512, activation=F.relu, bias=0.1
    ):
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.n_output_channels = n_output_channels
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(n_input_channels, 32, 8, stride=4),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.Conv2d(64, 64, 3, stride=1),
            ]
        )
        self.output = nn.Linear(3136, n_output_channels)

        self.apply(init_chainer_default)
        self.apply(constant_bias_initializer(bias=bias))

    def forward(self, state):
        h = state
        for layer in self.layers:
            h = self.activation(layer(h))
        h_flat = h.view(h.size(0), -1)
        return self.activation(self.output(h_flat))


class SmallAtariCNN(nn.Module):
    """Small CNN module proposed for DQN in NeurIPS DL Workshop, 2013.

    See: https://arxiv.org/abs/1312.5602
    """

    def __init__(
        self, n_input_channels=4, n_output_channels=256, activation=F.relu, bias=0.1
    ):
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.n_output_channels = n_output_channels
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(n_input_channels, 16, 8, stride=4),
                nn.Conv2d(16, 32, 4, stride=2),
            ]
        )
        self.output = nn.Linear(2592, n_output_channels)

        self.apply(init_chainer_default)
        self.apply(constant_bias_initializer(bias=bias))

    def forward(self, state):
        h = state
        for layer in self.layers:
            h = self.activation(layer(h))
        h_flat = h.view(h.size(0), -1)
        return self.activation(self.output(h_flat))
