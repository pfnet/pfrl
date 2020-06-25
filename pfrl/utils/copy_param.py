import torch


def copy_param(target_link, source_link):
    """Copy parameters of a link to another link."""
    target_link.load_state_dict(source_link.state_dict())


def soft_copy_param(target_link, source_link, tau):
    """Soft-copy parameters of a link to another link."""
    target_dict = target_link.state_dict()
    source_dict = source_link.state_dict()
    for k, target_value in target_dict.items():
        source_value = source_dict[k]
        if source_value.dtype in [torch.float32, torch.float64, torch.float16]:
            assert target_value.shape == source_value.shape
            target_value.mul_(1 - tau)
            target_value.add_(tau * source_value)
        else:
            # Scalar type
            # Some modules such as BN has scalar value `num_batches_tracked`
            target_dict[k] = source_value


def copy_grad(target_link, source_link):
    """Copy gradients of a link to another link."""
    for target_param, source_param in zip(
        target_link.parameters(), source_link.parameters()
    ):
        assert target_param.shape == source_param.shape
        if source_param.grad is None:
            target_param.grad = None
        else:
            target_param.grad = source_param.grad.clone()


def synchronize_parameters(src, dst, method, tau=None):
    {
        "hard": lambda: copy_param(dst, src),
        "soft": lambda: soft_copy_param(dst, src, tau),
    }[method]()
