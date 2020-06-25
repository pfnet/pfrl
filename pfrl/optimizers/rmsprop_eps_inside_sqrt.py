import torch


# mypy somehow complains about `torch.optim.RMSprop` as of torch==1.5.0.
class RMSpropEpsInsideSqrt(torch.optim.RMSprop):  # type: ignore
    """torch.optim.RMSprop with eps inside sqrt."""

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("RMSprop does not support sparse gradients")
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["square_avg"] = torch.zeros_like(p.data)
                    if group["momentum"] > 0:
                        state["momentum_buffer"] = torch.zeros_like(p.data)
                    if group["centered"]:
                        state["grad_avg"] = torch.zeros_like(p.data)

                square_avg = state["square_avg"]
                alpha = group["alpha"]

                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad = grad.add(group["weight_decay"], p.data)

                square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)

                if group["centered"]:
                    grad_avg = state["grad_avg"]
                    grad_avg.mul_(alpha).add_(1 - alpha, grad)
                    avg = (
                        square_avg.addcmul(-1, grad_avg, grad_avg)
                        .add_(group["eps"])
                        .sqrt_()
                    )
                else:
                    avg = square_avg.add(group["eps"]).sqrt_()

                if group["momentum"] > 0:
                    buf = state["momentum_buffer"]
                    buf.mul_(group["momentum"]).addcdiv_(grad, avg)
                    p.data.add_(-group["lr"], buf)
                else:
                    p.data.addcdiv_(-group["lr"], grad, avg)

        return loss


class SharedRMSpropEpsInsideSqrt(RMSpropEpsInsideSqrt):
    """RMSpropEpsInsideSqrt with non-lazy state initialization."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                # State initialization

                if len(state) == 0:
                    state["step"] = 0
                    state["square_avg"] = torch.zeros_like(p.data)
                    if group["momentum"] > 0:
                        state["momentum_buffer"] = torch.zeros_like(p.data)
                    if group["centered"]:
                        state["grad_avg"] = torch.zeros_like(p.data)
