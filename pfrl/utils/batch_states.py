from typing import Any, Callable, Sequence

import torch
from torch.utils.data._utils.collate import default_collate


def _to_recursive(batched: Any, device: torch.device) -> Any:
    if isinstance(batched, torch.Tensor):
        return batched.to(device)
    elif isinstance(batched, list):
        return [x.to(device) for x in batched]
    elif isinstance(batched, tuple):
        return tuple(x.to(device) for x in batched)
    else:
        raise TypeError("Unsupported type of data")


def batch_states(
    states: Sequence[Any], device: torch.device, phi: Callable[[Any], Any]
) -> Any:
    """The default method for making batch of observations.

    Args:
        states (list): list of observations from an environment.
        device (module): CPU or GPU the data should be placed on
        phi (callable): Feature extractor applied to observations

    Return:
        the object which will be given as input to the model.
    """
    features = [phi(s) for s in states]
    # return concat_examples(features, device=device)
    return _to_recursive(default_collate(features), device)
