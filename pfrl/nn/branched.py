import torch


class Branched(torch.nn.Module):
    """Module that calls forward functions of child modules in parallel.

    When the `forward` method of this module is called, all the
    arguments are forwarded to each child module's `forward` method.

    The returned values from the child modules are returned as a tuple.

    Args:
        *modules: Child modules. Each module should be callable.
    """

    def __init__(self, *modules):
        super().__init__()
        self.child_modules = torch.nn.ModuleList(modules)

    def forward(self, *args, **kwargs):
        """Forward the arguments to the child modules.

        Args:
            *args, **kwargs: Any arguments forwarded to child modules.  Each
                child module should be able to accept the arguments.

        Returns:
            tuple: Tuple of the returned values from the child modules.
        """
        return tuple(mod(*args, **kwargs) for mod in self.child_modules)
