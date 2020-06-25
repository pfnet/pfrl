from torch import nn

from pfrl.nn.recurrent import Recurrent


class RecurrentBranched(Recurrent, nn.ModuleList):
    """Recurrent module that bundles parallel branches.

    This is a recurrent analog to `pfrl.nn.Branched`. It bundles
    multiple recurrent modules.

    Args:
        *modules: Child modules. Each module should be recurrent and callable.
    """

    def __init__(self, *modules):
        super().__init__(modules)

    def forward(self, sequences, recurrent_state):
        if recurrent_state is None:
            n = len(self)
            recurrent_state = [None] * n
        child_ys, rs = tuple(
            zip(*[link(sequences, rs) for link, rs in zip(self, recurrent_state)])
        )
        return child_ys, rs
