import torch.nn as nn
import torch.nn.functional as F

from pfrl.initializers import init_lecun_normal


class LinearBN(nn.Module):
    """Linear layer with BatchNormalization."""

    def __init__(self, in_size, out_size):
        super().__init__()
        self.linear = nn.Linear(in_size, out_size)
        self.bn = nn.BatchNorm1d(out_size)

    def forward(self, x):
        return self.bn(self.linear(x))


class MLPBN(nn.Module):
    """Multi-Layer Perceptron with Batch Normalization.

    Args:
        in_size (int): Input size.
        out_size (int): Output size.
        hidden_sizes (list of ints): Sizes of hidden channels.
        normalize_input (bool): If set to True, Batch Normalization is applied
            to inputs.
        normalize_output (bool): If set to True, Batch Normalization is applied
            to outputs.
        nonlinearity (callable): Nonlinearity between layers. It must accept a
            Variable as an argument and return a Variable with the same shape.
            Nonlinearities with learnable parameters such as PReLU are not
            supported.
        last_wscale (float): Scale of weight initialization of the last layer.
    """

    def __init__(
        self,
        in_size,
        out_size,
        hidden_sizes,
        normalize_input=True,
        normalize_output=False,
        nonlinearity=F.relu,
        last_wscale=1,
    ):
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_sizes = hidden_sizes
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output
        self.nonlinearity = nonlinearity

        super().__init__()
        if normalize_input:
            self.input_bn = nn.BatchNorm1d(in_size)

        if hidden_sizes:
            self.hidden_layers = nn.ModuleList()
            self.hidden_layers.append(LinearBN(in_size, hidden_sizes[0]))
            for hin, hout in zip(hidden_sizes, hidden_sizes[1:]):
                self.hidden_layers.append(LinearBN(hin, hout))
            self.output = nn.Linear(hidden_sizes[-1], out_size)
        else:
            self.output = nn.Linear(in_size, out_size)
        init_lecun_normal(self.output.weight, scale=last_wscale)

        if normalize_output:
            self.output_bn = nn.BatchNorm1d(out_size)

    def forward(self, x):
        h = x
        if self.normalize_input:
            h = self.input_bn(h)
        if self.hidden_sizes:
            for layer in self.hidden_layers:
                h = self.nonlinearity(layer(h))
        h = self.output(h)
        if self.normalize_output:
            h = self.output_bn(h)
        return h
