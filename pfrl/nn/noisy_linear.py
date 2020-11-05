import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_lecun_uniform(tensor, scale=1.0):
    """Initializes the tensor with LeCunUniform."""
    fan_in = torch.nn.init._calculate_correct_fan(tensor, "fan_in")
    s = scale * np.sqrt(3.0 / fan_in)
    with torch.no_grad():
        return tensor.uniform_(-s, s)


def init_variance_scaling_constant(tensor, scale=1.0):

    if tensor.ndim == 1:
        s = scale / np.sqrt(tensor.shape[0])
    else:
        fan_in = torch.nn.init._calculate_correct_fan(tensor, "fan_in")
        s = scale / np.sqrt(fan_in)
    with torch.no_grad():
        return tensor.fill_(s)


class FactorizedNoisyLinear(nn.Module):
    """Linear layer in Factorized Noisy Network

    Args:
        mu_link (nn.Linear): Linear link that computes mean of output.
        sigma_scale (float): The hyperparameter sigma_0 in the original paper.
            Scaling factor of the initial weights of noise-scaling parameters.
    """

    def __init__(self, mu_link, sigma_scale=0.4):
        super(FactorizedNoisyLinear, self).__init__()
        self._kernel = None
        self.out_size = mu_link.out_features
        self.hasbias = mu_link.bias is not None

        in_size = mu_link.weight.shape[1]
        device = mu_link.weight.device
        self.mu = nn.Linear(in_size, self.out_size, bias=self.hasbias)
        init_lecun_uniform(self.mu.weight, scale=1 / np.sqrt(3))

        self.sigma = nn.Linear(in_size, self.out_size, bias=self.hasbias)
        init_variance_scaling_constant(self.sigma.weight, scale=sigma_scale)
        if self.hasbias:
            init_variance_scaling_constant(self.sigma.bias, scale=sigma_scale)

        self.mu.to(device)
        self.sigma.to(device)

    def _eps(self, shape, dtype, device):
        r = torch.normal(mean=0.0, std=1.0, size=(shape,), dtype=dtype, device=device)
        return torch.abs(torch.sqrt(torch.abs(r))) * torch.sign(r)

    def forward(self, x):
        # use info of sigma.W to avoid strange error messages
        dtype = self.sigma.weight.dtype
        out_size, in_size = self.sigma.weight.shape

        eps = self._eps(in_size + out_size, dtype, self.sigma.weight.device)
        eps_x = eps[:in_size]
        eps_y = eps[in_size:]
        W = torch.addcmul(self.mu.weight, self.sigma.weight, torch.ger(eps_y, eps_x))
        if self.hasbias:
            b = torch.addcmul(self.mu.bias, self.sigma.bias, eps_y)
            return F.linear(x, W, b)
        else:
            return F.linear(x, W)
