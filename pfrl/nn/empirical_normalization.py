import numpy as np
import torch
from torch import nn


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values.

    Args:
        shape (int or tuple of int): Shape of input values except batch axis.
        batch_axis (int): Batch axis.
        eps (float): Small value for stability.
        dtype (dtype): Dtype of input values.
        until (int or None): If this arg is specified, the link learns input
            values until the sum of batch sizes exceeds it.
    """

    def __init__(
        self,
        shape,
        batch_axis=0,
        eps=1e-2,
        dtype=np.float32,
        until=None,
        clip_threshold=None,
    ):
        super(EmpiricalNormalization, self).__init__()
        dtype = np.dtype(dtype)
        self.batch_axis = batch_axis
        self.eps = dtype.type(eps)
        self.until = until
        self.clip_threshold = clip_threshold
        self.register_buffer(
            "_mean",
            torch.tensor(np.expand_dims(np.zeros(shape, dtype=dtype), batch_axis)),
        )
        self.register_buffer(
            "_var",
            torch.tensor(np.expand_dims(np.ones(shape, dtype=dtype), batch_axis)),
        )
        self.register_buffer("count", torch.tensor(0))

        # cache
        self._cached_std_inverse = None

    @property
    def mean(self):
        return torch.squeeze(self._mean, self.batch_axis).clone()

    @property
    def std(self):
        return torch.sqrt(torch.squeeze(self._var, self.batch_axis)).clone()

    @property
    def _std_inverse(self):
        if self._cached_std_inverse is None:
            self._cached_std_inverse = (self._var + self.eps) ** -0.5

        return self._cached_std_inverse

    def experience(self, x):
        """Learn input values without computing the output values of them"""

        if self.until is not None and self.count >= self.until:
            return

        count_x = x.shape[self.batch_axis]
        if count_x == 0:
            return

        self.count += count_x
        rate = count_x / self.count.float()
        assert rate > 0
        assert rate <= 1

        var_x, mean_x = torch.var_mean(
            x, axis=self.batch_axis, keepdims=True, unbiased=False
        )
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))

        # clear cache
        self._cached_std_inverse = None

    def forward(self, x, update=True):
        """Normalize mean and variance of values based on emprical values.

        Args:
            x (ndarray or Variable): Input values
            update (bool): Flag to learn the input values

        Returns:
            ndarray or Variable: Normalized output values
        """

        if update:
            self.experience(x)

        normalized = (x - self._mean) * self._std_inverse
        if self.clip_threshold is not None:
            normalized = torch.clamp(
                normalized, -self.clip_threshold, self.clip_threshold
            )
        return normalized

    def inverse(self, y):
        std = torch.sqrt(self._var + self.eps)
        return y * std + self._mean
