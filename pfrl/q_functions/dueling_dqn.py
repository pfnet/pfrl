import torch
import torch.nn as nn
import torch.nn.functional as F

from pfrl import action_value
from pfrl.initializers import init_chainer_default
from pfrl.nn.mlp import MLP
from pfrl.q_function import StateQFunction


def constant_bias_initializer(bias=0.0):
    @torch.no_grad()
    def init_bias(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            m.bias.fill_(bias)

    return init_bias


class DuelingDQN(nn.Module, StateQFunction):
    """Dueling Q-Network

    See: http://arxiv.org/abs/1511.06581
    """

    def __init__(self, n_actions, n_input_channels=4, activation=F.relu, bias=0.1):
        self.n_actions = n_actions
        self.n_input_channels = n_input_channels
        self.activation = activation

        super().__init__()
        self.conv_layers = nn.ModuleList(
            [
                nn.Conv2d(n_input_channels, 32, 8, stride=4),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.Conv2d(64, 64, 3, stride=1),
            ]
        )

        self.a_stream = MLP(3136, n_actions, [512])
        self.v_stream = MLP(3136, 1, [512])

        self.conv_layers.apply(init_chainer_default)  # MLP already applies
        self.conv_layers.apply(constant_bias_initializer(bias=bias))

    def forward(self, x):
        h = x
        for layer in self.conv_layers:
            h = self.activation(layer(h))

        # Advantage
        batch_size = x.shape[0]
        h = h.reshape(batch_size, -1)
        ya = self.a_stream(h)
        mean = torch.reshape(torch.sum(ya, dim=1) / self.n_actions, (batch_size, 1))
        ya, mean = torch.broadcast_tensors(ya, mean)
        ya -= mean

        # State value
        ys = self.v_stream(h)

        ya, ys = torch.broadcast_tensors(ya, ys)
        q = ya + ys
        return action_value.DiscreteActionValue(q)


class DistributionalDuelingDQN(nn.Module, StateQFunction):
    """Distributional dueling fully-connected Q-function with discrete actions."""

    def __init__(
        self,
        n_actions,
        n_atoms,
        v_min,
        v_max,
        n_input_channels=4,
        activation=torch.relu,
        bias=0.1,
    ):
        assert n_atoms >= 2
        assert v_min < v_max

        self.n_actions = n_actions
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.n_atoms = n_atoms

        super().__init__()
        self.z_values = torch.linspace(v_min, v_max, n_atoms, dtype=torch.float32)

        self.conv_layers = nn.ModuleList(
            [
                nn.Conv2d(n_input_channels, 32, 8, stride=4),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.Conv2d(64, 64, 3, stride=1),
            ]
        )

        self.main_stream = nn.Linear(3136, 1024)
        self.a_stream = nn.Linear(512, n_actions * n_atoms)
        self.v_stream = nn.Linear(512, n_atoms)

        self.apply(init_chainer_default)
        self.conv_layers.apply(constant_bias_initializer(bias=bias))

    def forward(self, x):
        h = x
        for layer in self.conv_layers:
            h = self.activation(layer(h))

        # Advantage
        batch_size = x.shape[0]

        h = self.activation(self.main_stream(h.view(batch_size, -1)))
        h_a, h_v = torch.chunk(h, 2, dim=1)
        ya = self.a_stream(h_a).reshape((batch_size, self.n_actions, self.n_atoms))

        mean = ya.sum(dim=1, keepdim=True) / self.n_actions

        ya, mean = torch.broadcast_tensors(ya, mean)
        ya -= mean

        # State value
        ys = self.v_stream(h_v).reshape((batch_size, 1, self.n_atoms))
        ya, ys = torch.broadcast_tensors(ya, ys)
        q = F.softmax(ya + ys, dim=2)

        self.z_values = self.z_values.to(x.device)
        return action_value.DistributionalDiscreteActionValue(q, self.z_values)
