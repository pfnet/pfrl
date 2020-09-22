from torch import nn


class ConstantsMult(nn.Module):
    def __init__(self, constants):
        super().__init__()
        self.constants = constants

    def forward(self, x):
        return self.constants * x
