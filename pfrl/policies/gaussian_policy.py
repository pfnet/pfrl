import numpy as np
import torch
from torch import nn


class GaussianHeadWithStateIndependentCovariance(nn.Module):
    """Gaussian head with state-independent learned covariance.

    This link is intended to be attached to a neural network that outputs
    the mean of a Gaussian policy. The only learnable parameter this link has
    determines the variance in a state-independent way.

    State-independent parameterization of the variance of a Gaussian policy
    is often used with PPO and TRPO, e.g., in https://arxiv.org/abs/1709.06560.

    Args:
        action_size (int): Number of dimensions of the action space.
        var_type (str): Type of parameterization of variance. It must be
            'spherical' or 'diagonal'.
        var_func (callable): Callable that computes the variance from the var
            parameter. It should always return positive values.
        var_param_init (float): Initial value the var parameter.
    """

    def __init__(
        self,
        action_size,
        var_type="spherical",
        var_func=nn.functional.softplus,
        var_param_init=0,
    ):
        super().__init__()

        self.var_func = var_func
        var_size = {"spherical": 1, "diagonal": action_size}[var_type]

        self.var_param = nn.Parameter(
            torch.tensor(np.broadcast_to(var_param_init, var_size), dtype=torch.float,)
        )

    def forward(self, mean):
        """Return a Gaussian with given mean.

        Args:
            mean (torch.Tensor or ndarray): Mean of Gaussian.

        Returns:
            torch.distributions.Distribution: Gaussian whose mean is the
                mean argument and whose variance is computed from the parameter
                of this link.
        """
        var = self.var_func(self.var_param)
        return torch.distributions.Independent(
            torch.distributions.Normal(loc=mean, scale=torch.sqrt(var)), 1
        )


class GaussianHeadWithDiagonalCovariance(nn.Module):
    """Gaussian head with diagonal covariance.

    This module is intended to be attached to a neural network that outputs
    a vector that is twice the size of an action vector. The vector is split
    and interpreted as the mean and diagonal covariance of a Gaussian policy.

    Args:
        var_func (callable): Callable that computes the variance
            from the second input. It should always return positive values.
    """

    def __init__(self, var_func=nn.functional.softplus):
        super().__init__()
        self.var_func = var_func

    def forward(self, mean_and_var):
        """Return a Gaussian with given mean and diagonal covariance.

        Args:
            mean_and_var (torch.Tensor): Vector that is twice the size of an
                action vector.

        Returns:
            torch.distributions.Distribution: Gaussian distribution with given
                mean and diagonal covariance.
        """
        assert mean_and_var.ndim == 2
        mean, pre_var = mean_and_var.chunk(2, dim=1)
        scale = self.var_func(pre_var).sqrt()
        return torch.distributions.Independent(
            torch.distributions.Normal(loc=mean, scale=scale), 1
        )


class GaussianHeadWithFixedCovariance(nn.Module):
    """Gaussian head with fixed covariance.

    This module is intended to be attached to a neural network that outputs
    the mean of a Gaussian policy. Its covariance is fixed to a diagonal matrix
    with a given scale.

    Args:
        scale (float): Scale parameter.
    """

    def __init__(self, scale=1):
        super().__init__()
        self.scale = scale

    def forward(self, mean):
        """Return a Gaussian with given mean.

        Args:
            mean (torch.Tensor): Batch of mean vectors.

        Returns:
            torch.distributions.Distribution: Multivariate Gaussian whose mean
                is the mean argument and whose scale is fixed.
        """
        return torch.distributions.Independent(
            torch.distributions.Normal(loc=mean, scale=self.scale), 1
        )
