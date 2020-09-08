import torch


def bound_by_tanh(x, low, high):
    """Bound a given value into [low, high] by tanh.

    Args:
        x (torch.Tensor): value to bound
        low (numpy.ndarray): lower bound
        high (numpy.ndarray): upper bound

    Returns:
        torch.Tensor: bounded value
    """
    assert isinstance(x, torch.Tensor)
    assert low is not None
    assert high is not None
    low = torch.as_tensor(low, dtype=x.dtype, device=x.device)
    high = torch.as_tensor(high, dtype=x.dtype, device=x.device)
    scale = (high - low) / 2
    loc = (high + low) / 2
    return torch.tanh(x) * scale + loc
