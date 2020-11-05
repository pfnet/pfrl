from torch import nn

from pfrl.nn.recurrent import Recurrent
from pfrl.utils.recurrent import (
    get_packed_sequence_info,
    is_recurrent,
    unwrap_packed_sequences_recursive,
    wrap_packed_sequences_recursive,
)


class RecurrentSequential(Recurrent, nn.Sequential):
    """Sequential model that can contain stateless recurrent modules.

    This is a recurrent analog to `torch.nn.Sequential`. It supports
    the recurrent interface by automatically detecting recurrent
    modules and handles recurrent states properly.

    For non-recurrent layers, this module automatically concatenates
    the input to the layers for efficient computation.

    Args:
        *layers: Callable objects.
    """

    def forward(self, sequences, recurrent_state):
        if recurrent_state is None:
            recurrent_state_queue = [None] * len(self.recurrent_children)
        else:
            assert len(recurrent_state) == len(self.recurrent_children)
            recurrent_state_queue = list(reversed(recurrent_state))
        new_recurrent_state = []
        h = sequences
        batch_sizes, sorted_indices = get_packed_sequence_info(h)
        is_wrapped = True
        for layer in self:
            if is_recurrent(layer):
                if not is_wrapped:
                    h = wrap_packed_sequences_recursive(h, batch_sizes, sorted_indices)
                    is_wrapped = True
                rs = recurrent_state_queue.pop()
                h, rs = layer(h, rs)
                new_recurrent_state.append(rs)
            else:
                if is_wrapped:
                    h = unwrap_packed_sequences_recursive(h)
                    is_wrapped = False
                h = layer(h)
        if not is_wrapped:
            h = wrap_packed_sequences_recursive(h, batch_sizes, sorted_indices)
        assert not recurrent_state_queue
        assert len(new_recurrent_state) == len(self.recurrent_children)
        return h, tuple(new_recurrent_state)

    @property
    def recurrent_children(self):
        """Return recurrent child modules.

        Returns:
            tuple: Child modules that are recurrent.
        """
        return tuple(child for child in self if is_recurrent(child))
