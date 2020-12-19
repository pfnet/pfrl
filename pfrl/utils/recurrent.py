import itertools

import numpy as np
import torch
from torch import nn


def is_recurrent(layer):
    """Return True iff a given layer is recurrent and supported by PFRL.

    Args:
        layer (callable): Any callable object.

    Returns:
        bool: True iff a given layer is recurrent and supported by PFRL.
    """
    # Import here to avoid circular import
    from pfrl.nn import Recurrent

    return isinstance(
        layer,
        (
            nn.LSTM,
            nn.RNN,
            nn.GRU,
            Recurrent,
        ),
    )


def mask_recurrent_state_at(recurrent_state, indices):
    """Return a recurrent state masked at given indices.

    This function can be used to initialize a recurrent state only for a
    certain sequence, not all the sequences.

    Args:
        recurrent_state (object): Batched recurrent state.
        indices (int or array-like of ints): Which recurrent state to mask.

    Returns:
        object: New batched recurrent state.
    """
    if recurrent_state is None:
        return None
    elif isinstance(recurrent_state, torch.Tensor):
        mask = torch.ones_like(recurrent_state)
        mask[:, indices] = 0
        return recurrent_state * mask
    elif isinstance(recurrent_state, tuple):
        return tuple(mask_recurrent_state_at(s, indices) for s in recurrent_state)
    else:
        raise ValueError("Invalid recurrent state: {}".format(recurrent_state))


def get_recurrent_state_at(recurrent_state, indices, detach):
    """Get a recurrent state at given indices.

    This function can be used to save a recurrent state so that you can
    reuse it when you replay past sequences.

    Args:
        indices (int or array-like of ints): Which recurrent state to get.

    Returns:
        object: Recurrent state of given indices.
    """
    if recurrent_state is None:
        return None
    elif isinstance(recurrent_state, torch.Tensor):
        if detach:
            recurrent_state = recurrent_state.detach()
        return recurrent_state[:, indices]
    elif isinstance(recurrent_state, tuple):
        return tuple(
            get_recurrent_state_at(s, indices, detach) for s in recurrent_state
        )
    else:
        raise ValueError("Invalid recurrent state: {}".format(recurrent_state))


def concatenate_recurrent_states(split_recurrent_states):
    """Concatenate recurrent states into a batch.

    This function can be used to make a batched recurrent state from separate
    recurrent states obtained via the `get_recurrent_state_at` function.

    Args:
        split_recurrent_states (Sequence): Recurrent states to concatenate.

    Returns:
        object: Batched recurrent_state.
    """
    if all(s is None for s in split_recurrent_states):
        return None
    else:
        non_none_s = next(s for s in split_recurrent_states if s is not None)
        if isinstance(non_none_s, torch.Tensor):
            new_ss = [
                s if s is not None else torch.zeros_like(non_none_s)
                for s in split_recurrent_states
            ]
            return torch.stack(new_ss, dim=1)
        elif isinstance(non_none_s, np.ndarray):
            new_ss = [
                s if s is not None else np.zeros_like(non_none_s)
                for s in split_recurrent_states
            ]
            return np.stack(new_ss, axis=1)
        elif isinstance(non_none_s, tuple):
            return tuple(
                concatenate_recurrent_states(
                    [s[i] if s is not None else None for s in split_recurrent_states]
                )
                for i in range(len(non_none_s))
            )
        else:
            raise ValueError("Invalid recurrent state: {}".format(non_none_s))


def pack_one_step_batch_as_sequences(xs):
    if isinstance(xs, tuple):
        return tuple(pack_one_step_batch_as_sequences(x) for x in xs)
    else:
        return nn.utils.rnn.pack_sequence(xs[:, None])


def unpack_sequences_as_one_step_batch(pack):
    if isinstance(pack, nn.utils.rnn.PackedSequence):
        return pack.data
    elif isinstance(pack, tuple):
        return tuple(unpack_sequences_as_one_step_batch(x) for x in pack)
    else:
        return pack


def one_step_forward(rnn, batch_input, recurrent_state):
    """One-step batch forward computation of a recurrent module.

    Args:
        rnn (torch.nn.Module): Recurrent module.
        batch_input (BatchData): One-step batched input.
        recurrent_state (object): Batched recurrent state.

    Returns:
        object: One-step batched output.
        object: New batched recurrent state.
    """
    pack = pack_one_step_batch_as_sequences(batch_input)
    y, recurrent_state = rnn(pack, recurrent_state)
    return unpack_sequences_as_one_step_batch(y), recurrent_state


def pack_and_forward(rnn, sequences, recurrent_state):
    """Pack sequences, multi-step forward, and then unwrap `PackedSequence`.

    Args:
        rnn (torch.nn.Module): Recurrent module.
        sequences (object): Sequences of input data.
        recurrent_state (object): Batched recurrent state.

    Returns:
        object: Sequence of output data, packed with time axis first.
        object: New batched recurrent state.
    """
    pack = pack_sequences_recursive(sequences)
    y, recurrent_state = rnn(pack, recurrent_state)
    return unwrap_packed_sequences_recursive(y), recurrent_state


def flatten_sequences_time_first(sequences):
    """Flatten sequences with time axis first.

    The resulting order is the same as how
    `torch.nn.utils.rnn.pack_sequence` will pack sequences into a tensor.

    Args:
        sequences: Sequences with batch axis first.

    Returns:
        list: Flattened sequences with time axis first.
    """
    ret = []
    for batch in itertools.zip_longest(*sequences):
        ret.extend([x for x in batch if x is not None])
    return ret


def wrap_packed_sequences_recursive(unwrapped, batch_sizes, sorted_indices):
    """Wrap packed tensors by `PackedSequence`.

    Args:
        unwrapped (object): Packed but unwrapped tensor(s).
        batch_sizes (Tensor): See `PackedSequence.batch_sizes`.
        sorted_indices (Tensor): See `PackedSequence.sorted_indices`.

    Returns:
        object: Packed sequences. If `unwrapped` is a tensor, then the returned
            value is a `PackedSequence`. If `unwrapped` is a tuple of tensors,
            then the returned value is a tuple of `PackedSequence`s.
    """
    if isinstance(unwrapped, torch.Tensor):
        return torch.nn.utils.rnn.PackedSequence(
            unwrapped, batch_sizes=batch_sizes, sorted_indices=sorted_indices
        )
    if isinstance(unwrapped, tuple):
        return tuple(
            wrap_packed_sequences_recursive(x, batch_sizes, sorted_indices)
            for x in unwrapped
        )
    return unwrapped


def unwrap_packed_sequences_recursive(packed):
    """Unwrap `PackedSequence` class of packed sequences recursively.

    This function extract `torch.Tensor` that
    `torch.nn.utils.rnn.PackedSequence` holds internally. Sequences in the
    internal tensor is ordered with time axis first.

    Unlike `torch.nn.pad_packed_sequence`, this function just returns the
    underlying tensor as it is without padding.

    To wrap the data by `PackedSequence` again, use
    `wrap_packed_sequences_recursive`.

    Args:
        packed (object): Packed sequences.

    Returns:
        object: Unwrapped packed sequences. If `packed` is a `PackedSequence`,
            then the returned value is `PackedSequence.data`, the underlying
            tensor. If `Packed` is a tuple of `PackedSequence`, then the
            returned value is a tuple of the underlying tensors.
    """
    if isinstance(packed, torch.nn.utils.rnn.PackedSequence):
        return packed.data
    if isinstance(packed, tuple):
        return tuple(unwrap_packed_sequences_recursive(x) for x in packed)
    return packed


def pack_sequences_recursive(sequences):
    """Pack sequences into PackedSequence recursively.

    This function works similarly to `torch.nn.utils.rnn.pack_sequence` except
    that it works recursively for tuples.

    When each given sequence is an N-tuple of `torch.Tensor`s, the function
    returns an N-tuple of `torch.nn.utils.rnn.PackedSequence`, packing i-th
    tensors separately for i=1,...,N.

    Args:
        sequences (object): Batch of sequences to pack.

    Returns:
        object: Packed sequences. If `sequences` is a list of tensors, then the
            returned value is a `PackedSequence`. If `sequences` is a list of
            tuples of tensors, then the returned value is a tuple of
            `PackedSequence`.
    """
    assert sequences
    first_seq = sequences[0]
    if isinstance(first_seq, torch.Tensor):
        return nn.utils.rnn.pack_sequence(sequences)
    if isinstance(first_seq, tuple):
        return tuple(
            pack_sequences_recursive([seq[i] for seq in sequences])
            for i in range(len(first_seq))
        )
    return sequences


def get_packed_sequence_info(packed):
    """Get `batch_sizes` and `sorted_indices` of `PackedSequence`.

    Args:
        packed (object): Packed sequences. If it contains multiple
            `PackedSequence`s, then only one of them are sampled assuming that
            all of them have same `batch_sizes` and `sorted_indices`.

    Returns:
        Tensor: `PackedSequence.batch_sizes`.
        Tensor: `PackedSequence.sorted_indices`.
    """
    if isinstance(packed, torch.nn.utils.rnn.PackedSequence):
        return packed.batch_sizes, packed.sorted_indices
    if isinstance(packed, tuple):
        for y in packed:
            ret = get_packed_sequence_info(y)
            if ret is not None:
                return ret
    return None


def recurrent_state_as_numpy(recurrent_state):
    """Convert a recurrent state in torch.Tensor to numpy.ndarray.

    Args:
        recurrent_state (object): Recurrent state in torch.Tensor.

    Returns:
        object: Recurrent state in numpy.ndarray.
    """
    if recurrent_state is None:
        return None
    elif isinstance(recurrent_state, torch.Tensor):
        return recurrent_state.detach().cpu().numpy()
    elif isinstance(recurrent_state, tuple):
        return tuple(recurrent_state_as_numpy(s) for s in recurrent_state)
    else:
        raise ValueError("Invalid recurrent state: {}".format(recurrent_state))


def recurrent_state_from_numpy(recurrent_state, device):
    """Convert a recurrent state in numpy.ndarray to torch.Tensor.

    Args:
        recurrent_state (object): Recurrent state in numpy.ndarray.
        device (torch.Device): Device the recurrent state is moved to.

    Returns:
        object: Recurrent state in torch.Tensor of a given device.
    """
    if recurrent_state is None:
        return None
    elif isinstance(recurrent_state, np.ndarray):
        return torch.from_numpy(recurrent_state).to(device)
    elif isinstance(recurrent_state, tuple):
        return tuple(recurrent_state_from_numpy(s, device) for s in recurrent_state)
    else:
        raise ValueError("Invalid recurrent state: {}".format(recurrent_state))


def detach_recurrent_state(recurrent_state):
    """Detach recurrent state.

    Args:
        recurrent_state (object): Recurrent state in torch.Tensor.

    Returns:
        object: Detached recurrent state.
    """
    if recurrent_state is None:
        return
    elif isinstance(recurrent_state, torch.Tensor):
        return recurrent_state.detach()
    elif isinstance(recurrent_state, tuple):
        return tuple(detach_recurrent_state(s) for s in recurrent_state)
    else:
        raise ValueError("Invalid recurrent state: {}".format(recurrent_state))
