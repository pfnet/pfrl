import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pfrl.action_value import (
    DiscreteActionValue,
    DistributionalDiscreteActionValue,
    QuadraticActionValue,
)
from pfrl.functions.lower_triangular_matrix import lower_triangular_matrix
from pfrl.initializers import init_chainer_default
from pfrl.nn import Lambda
from pfrl.nn.mlp import MLP
from pfrl.q_function import StateQFunction


def scale_by_tanh(x, low, high):
    scale = (high - low) / 2
    scale = torch.unsqueeze(torch.from_numpy(scale), dim=0).to(x.device)
    mean = (high + low) / 2
    mean = torch.unsqueeze(torch.from_numpy(mean), dim=0).to(x.device)
    return torch.tanh(x) * scale + mean


class SingleModelStateQFunctionWithDiscreteAction(nn.Module, StateQFunction):
    """Q-function with discrete actions.

    Args:
        model (nn.Module):
            Model that is callable and outputs action values.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        h = self.model(x)
        return DiscreteActionValue(h)


class FCStateQFunctionWithDiscreteAction(SingleModelStateQFunctionWithDiscreteAction):
    """Fully-connected state-input Q-function with discrete actions.

    Args:
        n_dim_obs: number of dimensions of observation space
        n_actions (int): Number of actions in action space.
        n_hidden_channels: number of hidden channels
        n_hidden_layers: number of hidden layers
        nonlinearity (callable): Nonlinearity applied after each hidden layer.
        last_wscale (float): Weight scale of the last layer.
    """

    def __init__(
        self,
        ndim_obs,
        n_actions,
        n_hidden_channels,
        n_hidden_layers,
        nonlinearity=F.relu,
        last_wscale=1.0,
    ):
        super().__init__(
            model=MLP(
                in_size=ndim_obs,
                out_size=n_actions,
                hidden_sizes=[n_hidden_channels] * n_hidden_layers,
                nonlinearity=nonlinearity,
                last_wscale=last_wscale,
            )
        )


class DistributionalSingleModelStateQFunctionWithDiscreteAction(
    nn.Module, StateQFunction
):
    """Distributional Q-function with discrete actions.

    Args:
        model (nn.Module):
            model that is callable and outputs atoms for each action.
        z_values (ndarray): Returns represented by atoms. Its shape must be
            (n_atoms,).
    """

    def __init__(self, model, z_values):
        super().__init__()
        self.model = model
        self.register_buffer("z_values", torch.from_numpy(z_values))

    def forward(self, x):
        h = self.model(x)
        return DistributionalDiscreteActionValue(h, self.z_values)


class DistributionalFCStateQFunctionWithDiscreteAction(
    DistributionalSingleModelStateQFunctionWithDiscreteAction
):
    """Distributional fully-connected Q-function with discrete actions.

    Args:
        n_dim_obs (int): Number of dimensions of observation space.
        n_actions (int): Number of actions in action space.
        n_atoms (int): Number of atoms of return distribution.
        v_min (float): Minimum value this model can approximate.
        v_max (float): Maximum value this model can approximate.
        n_hidden_channels (int): Number of hidden channels.
        n_hidden_layers (int): Number of hidden layers.
        nonlinearity (callable): Nonlinearity applied after each hidden layer.
        last_wscale (float): Weight scale of the last layer.
    """

    def __init__(
        self,
        ndim_obs,
        n_actions,
        n_atoms,
        v_min,
        v_max,
        n_hidden_channels,
        n_hidden_layers,
        nonlinearity=F.relu,
        last_wscale=1.0,
    ):
        assert n_atoms >= 2
        assert v_min < v_max
        z_values = np.linspace(v_min, v_max, num=n_atoms, dtype=np.float32)
        model = nn.Sequential(
            MLP(
                in_size=ndim_obs,
                out_size=n_actions * n_atoms,
                hidden_sizes=[n_hidden_channels] * n_hidden_layers,
                nonlinearity=nonlinearity,
                last_wscale=last_wscale,
            ),
            Lambda(lambda x: torch.reshape(x, (-1, n_actions, n_atoms))),
            nn.Softmax(dim=2),
        )
        super().__init__(model=model, z_values=z_values)


class FCQuadraticStateQFunction(nn.Module, StateQFunction):
    """Fully-connected state-input continuous Q-function.

    See: https://arxiv.org/abs/1603.00748

    Args:
        n_input_channels: number of input channels
        n_dim_action: number of dimensions of action space
        n_hidden_channels: number of hidden channels
        n_hidden_layers: number of hidden layers
        action_space: action_space
        scale_mu (bool): scale mu by applying tanh if True
    """

    def __init__(
        self,
        n_input_channels,
        n_dim_action,
        n_hidden_channels,
        n_hidden_layers,
        action_space,
        scale_mu=True,
    ):
        self.n_input_channels = n_input_channels
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels
        self.n_dim_action = n_dim_action
        assert action_space is not None
        self.scale_mu = scale_mu
        self.action_space = action_space
        super().__init__()
        hidden_layers = nn.ModuleList()
        assert n_hidden_layers >= 1
        hidden_layers.append(
            init_chainer_default(nn.Linear(n_input_channels, n_hidden_channels))
        )
        for _ in range(n_hidden_layers - 1):
            hidden_layers.append(
                init_chainer_default(nn.Linear(n_hidden_channels, n_hidden_channels))
            )
        self.hidden_layers = hidden_layers

        self.v = init_chainer_default(nn.Linear(n_hidden_channels, 1))
        self.mu = init_chainer_default(nn.Linear(n_hidden_channels, n_dim_action))
        self.mat_diag = init_chainer_default(nn.Linear(n_hidden_channels, n_dim_action))
        non_diag_size = n_dim_action * (n_dim_action - 1) // 2
        if non_diag_size > 0:
            self.mat_non_diag = init_chainer_default(
                nn.Linear(n_hidden_channels, non_diag_size)
            )

    def forward(self, state):
        h = state
        for layer in self.hidden_layers:
            h = F.relu(layer(h))
        v = self.v(h)
        mu = self.mu(h)

        if self.scale_mu:
            mu = scale_by_tanh(
                mu, high=self.action_space.high, low=self.action_space.low
            )

        mat_diag = torch.exp(self.mat_diag(h))
        if hasattr(self, "mat_non_diag"):
            mat_non_diag = self.mat_non_diag(h)
            tril = lower_triangular_matrix(mat_diag, mat_non_diag)
            mat = torch.matmul(tril, torch.transpose(tril, 1, 2))
        else:
            mat = torch.unsqueeze(mat_diag ** 2, dim=2)
        return QuadraticActionValue(
            mu,
            mat,
            v,
            min_action=self.action_space.low,
            max_action=self.action_space.high,
        )


class DiscreteActionValueHead(nn.Module):
    def forward(self, q_values):
        return DiscreteActionValue(q_values)
