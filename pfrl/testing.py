import numpy as np
import torch


def _as_numpy_recursive(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, list) or isinstance(x, tuple):
        return np.asarray([_as_numpy_recursive(y) for y in x])
    return x


def torch_assert_allclose(actual, desired, *args, **kwargs):
    """Assert two objects are equal up to desired tolerance.

    This function can be used as a replacement of
    `numpy.testing.assert_allclose` except that lists, tuples, and
    `torch.Tensor`s are converted to `numpy.ndarray`s automatically before
    comparison.
    """
    actual = _as_numpy_recursive(actual)
    desired = _as_numpy_recursive(desired)
    np.testing.assert_allclose(actual, desired, *args, **kwargs)
