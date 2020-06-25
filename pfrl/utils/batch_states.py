import torch


def _to_recursive(batched, device):
    if isinstance(batched, torch.Tensor):
        return batched.to(device)
    elif isinstance(batched, list):
        return [x.to(device) for x in batched]
    elif isinstance(batched, tuple):
        return tuple(x.to(device) for x in batched)
    else:
        raise TypeError("Unsupported type of data")


def batch_states(states, device, phi):
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
    return _to_recursive(
        torch.utils.data._utils.collate.default_collate(features), device
    )
