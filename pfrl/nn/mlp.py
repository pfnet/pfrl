import torch.nn as nn
import torch.nn.functional as F

from pfrl.initializers import init_chainer_default, init_lecun_normal


class MLP(nn.Module):
    """Multi-Layer Perceptron"""

    def __init__(
        self, in_size, out_size, hidden_sizes, nonlinearity=F.relu, last_wscale=1
    ):
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        super().__init__()
        if hidden_sizes:
            self.hidden_layers = nn.ModuleList()
            self.hidden_layers.append(nn.Linear(in_size, hidden_sizes[0]))
            for hin, hout in zip(hidden_sizes, hidden_sizes[1:]):
                self.hidden_layers.append(nn.Linear(hin, hout))
            self.hidden_layers.apply(init_chainer_default)
            self.output = nn.Linear(hidden_sizes[-1], out_size)
        else:
            self.output = nn.Linear(in_size, out_size)

        init_lecun_normal(self.output.weight, scale=last_wscale)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        h = x
        if self.hidden_sizes:
            for layer in self.hidden_layers:
                h = self.nonlinearity(layer(h))
        return self.output(h)
