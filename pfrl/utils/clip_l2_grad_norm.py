import numpy as np
import torch


def clip_l2_grad_norm_(parameters, max_norm):
    """Clip gradient L2 norm.

    This function works in the same way as `torch.nn.utils.clip_grad_norm_`
    with `norm_type=2`, but more efficiently on CPU as of PyTorch 1.4.0.

    Args:
        parameters (torch.Tensor or Iterable[torch.Tensor]): `torch.Tensor`(s)
            that will have gradients normalized.
        max_norm (float or int): Maximum norm of the gradients.

    Returns:
        float: L2 norm of the unclipped gradient.

    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        parameters = list(parameters)
    if not parameters:
        return 0
    if parameters[0].is_cuda:
        # On GPU, `torch.nn.utils.clip_grad_norm_` is fast enough
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    total_norm = np.linalg.norm(
        [np.linalg.norm(p.grad.detach().cpu().numpy()) for p in parameters]
    )
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef)
    return total_norm
