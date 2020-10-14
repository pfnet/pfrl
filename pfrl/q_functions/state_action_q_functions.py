import torch
import torch.nn as nn
import torch.nn.functional as F

from pfrl.initializers import init_lecun_normal
from pfrl.nn.mlp import MLP
from pfrl.nn.mlp_bn import MLPBN
from pfrl.q_function import StateActionQFunction


class SingleModelStateActionQFunction(nn.Module, StateActionQFunction):
    """Q-function with discrete actions.

    Args:
        model (nn.Module):
            Module that is callable and outputs action values.
    """

    def __init__(self, model):
        super().__init__(model=model)

    def forward(self, x, a):
        h = self.model(x, a)
        return h


class FCSAQFunction(MLP, StateActionQFunction):
    """Fully-connected (s,a)-input Q-function.

    Args:
        n_dim_obs (int): Number of dimensions of observation space.
        n_dim_action (int): Number of dimensions of action space.
        n_hidden_channels (int): Number of hidden channels.
        n_hidden_layers (int): Number of hidden layers.
        nonlinearity (callable): Nonlinearity between layers. It must accept a
            Variable as an argument and return a Variable with the same shape.
            Nonlinearities with learnable parameters such as PReLU are not
            supported. It is not used if n_hidden_layers is zero.
        last_wscale (float): Scale of weight initialization of the last layer.
    """

    def __init__(
        self,
        n_dim_obs,
        n_dim_action,
        n_hidden_channels,
        n_hidden_layers,
        nonlinearity=F.relu,
        last_wscale=1.0,
    ):
        self.n_input_channels = n_dim_obs + n_dim_action
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels
        self.nonlinearity = nonlinearity
        super().__init__(
            in_size=self.n_input_channels,
            out_size=1,
            hidden_sizes=[self.n_hidden_channels] * self.n_hidden_layers,
            nonlinearity=nonlinearity,
            last_wscale=last_wscale,
        )

    def forward(self, state, action):
        h = torch.cat((state, action), dim=1)
        return super().forward(h)


class FCLSTMSAQFunction(nn.Module, StateActionQFunction):
    """Fully-connected + LSTM (s,a)-input Q-function.

    Args:
        n_dim_obs (int): Number of dimensions of observation space.
        n_dim_action (int): Number of dimensions of action space.
        n_hidden_channels (int): Number of hidden channels.
        n_hidden_layers (int): Number of hidden layers.
        nonlinearity (callable): Nonlinearity between layers. It must accept a
            Variable as an argument and return a Variable with the same shape.
            Nonlinearities with learnable parameters such as PReLU are not
            supported.
        last_wscale (float): Scale of weight initialization of the last layer.
    """

    def __init__(
        self,
        n_dim_obs,
        n_dim_action,
        n_hidden_channels,
        n_hidden_layers,
        nonlinearity=F.relu,
        last_wscale=1.0,
    ):
        raise NotImplementedError()
        self.n_input_channels = n_dim_obs + n_dim_action
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels
        self.nonlinearity = nonlinearity
        super().__init__()
        self.fc = MLP(
            self.n_input_channels,
            n_hidden_channels,
            [self.n_hidden_channels] * self.n_hidden_layers,
            nonlinearity=nonlinearity,
        )
        self.lstm = nn.LSTM(
            num_layers=1, input_size=n_hidden_channels, hidden_size=n_hidden_channels
        )
        self.out = nn.Linear(n_hidden_channels, 1)
        for (n, p) in self.lstm.named_parameters():
            if "weight" in n:
                init_lecun_normal(p)
            else:
                nn.init.zeros_(p)

        init_lecun_normal(self.out.weight, scale=last_wscale)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, a):
        h = torch.cat((x, a), dim=1)
        h = self.nonlinearity(self.fc(h))
        h = self.lstm(h)
        return self.out(h)


class FCBNSAQFunction(MLPBN, StateActionQFunction):
    """Fully-connected + BN (s,a)-input Q-function.

    Args:
        n_dim_obs (int): Number of dimensions of observation space.
        n_dim_action (int): Number of dimensions of action space.
        n_hidden_channels (int): Number of hidden channels.
        n_hidden_layers (int): Number of hidden layers.
        normalize_input (bool): If set to True, Batch Normalization is applied
            to both observations and actions.
        nonlinearity (callable): Nonlinearity between layers. It must accept a
            Variable as an argument and return a Variable with the same shape.
            Nonlinearities with learnable parameters such as PReLU are not
            supported. It is not used if n_hidden_layers is zero.
        last_wscale (float): Scale of weight initialization of the last layer.
    """

    def __init__(
        self,
        n_dim_obs,
        n_dim_action,
        n_hidden_channels,
        n_hidden_layers,
        normalize_input=True,
        nonlinearity=F.relu,
        last_wscale=1.0,
    ):
        self.n_input_channels = n_dim_obs + n_dim_action
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels
        self.normalize_input = normalize_input
        self.nonlinearity = nonlinearity
        super().__init__(
            in_size=self.n_input_channels,
            out_size=1,
            hidden_sizes=[self.n_hidden_channels] * self.n_hidden_layers,
            normalize_input=self.normalize_input,
            nonlinearity=nonlinearity,
            last_wscale=last_wscale,
        )

    def forward(self, state, action):
        h = torch.cat((state, action), dim=1)
        return super().forward(h)


class FCBNLateActionSAQFunction(nn.Module, StateActionQFunction):
    """Fully-connected + BN (s,a)-input Q-function with late action input.

    Actions are not included until the second hidden layer and not normalized.
    This architecture is used in the DDPG paper:
    http://arxiv.org/abs/1509.02971

    Args:
        n_dim_obs (int): Number of dimensions of observation space.
        n_dim_action (int): Number of dimensions of action space.
        n_hidden_channels (int): Number of hidden channels.
        n_hidden_layers (int): Number of hidden layers. It must be greater than
            or equal to 1.
        normalize_input (bool): If set to True, Batch Normalization is applied
        nonlinearity (callable): Nonlinearity between layers. It must accept a
            Variable as an argument and return a Variable with the same shape.
            Nonlinearities with learnable parameters such as PReLU are not
            supported.
        last_wscale (float): Scale of weight initialization of the last layer.
    """

    def __init__(
        self,
        n_dim_obs,
        n_dim_action,
        n_hidden_channels,
        n_hidden_layers,
        normalize_input=True,
        nonlinearity=F.relu,
        last_wscale=1.0,
    ):
        assert n_hidden_layers >= 1
        self.n_input_channels = n_dim_obs + n_dim_action
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels
        self.normalize_input = normalize_input
        self.nonlinearity = nonlinearity

        super().__init__()
        # No need to pass nonlinearity to obs_mlp because it has no
        # hidden layers
        self.obs_mlp = MLPBN(
            in_size=n_dim_obs,
            out_size=n_hidden_channels,
            hidden_sizes=[],
            normalize_input=normalize_input,
            normalize_output=True,
        )
        self.mlp = MLP(
            in_size=n_hidden_channels + n_dim_action,
            out_size=1,
            hidden_sizes=([self.n_hidden_channels] * (self.n_hidden_layers - 1)),
            nonlinearity=nonlinearity,
            last_wscale=last_wscale,
        )

        self.output = self.mlp.output

    def forward(self, state, action):
        h = self.nonlinearity(self.obs_mlp(state))
        h = torch.cat((h, action), dim=1)
        return self.mlp(h)


class FCLateActionSAQFunction(nn.Module, StateActionQFunction):
    """Fully-connected (s,a)-input Q-function with late action input.

    Actions are not included until the second hidden layer and not normalized.
    This architecture is used in the DDPG paper:
    http://arxiv.org/abs/1509.02971

    Args:
        n_dim_obs (int): Number of dimensions of observation space.
        n_dim_action (int): Number of dimensions of action space.
        n_hidden_channels (int): Number of hidden channels.
        n_hidden_layers (int): Number of hidden layers. It must be greater than
            or equal to 1.
        nonlinearity (callable): Nonlinearity between layers. It must accept a
            Variable as an argument and return a Variable with the same shape.
            Nonlinearities with learnable parameters such as PReLU are not
            supported.
        last_wscale (float): Scale of weight initialization of the last layer.
    """

    def __init__(
        self,
        n_dim_obs,
        n_dim_action,
        n_hidden_channels,
        n_hidden_layers,
        nonlinearity=F.relu,
        last_wscale=1.0,
    ):
        assert n_hidden_layers >= 1
        self.n_input_channels = n_dim_obs + n_dim_action
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels
        self.nonlinearity = nonlinearity

        super().__init__()
        # No need to pass nonlinearity to obs_mlp because it has no
        # hidden layers
        self.obs_mlp = MLP(
            in_size=n_dim_obs, out_size=n_hidden_channels, hidden_sizes=[]
        )
        self.mlp = MLP(
            in_size=n_hidden_channels + n_dim_action,
            out_size=1,
            hidden_sizes=([self.n_hidden_channels] * (self.n_hidden_layers - 1)),
            nonlinearity=nonlinearity,
            last_wscale=last_wscale,
        )

        self.output = self.mlp.output

    def forward(self, state, action):
        h = self.nonlinearity(self.obs_mlp(state))
        h = torch.cat((h, action), dim=1)
        return self.mlp(h)
