class Recurrent(object):
    """Recurrent module interface.

    This class defines the interface of a recurrent module PFRL support.

    The interface is similar to that of `torch.nn.LSTM` except that sequential
    data are expected to be packed in `torch.nn.utils.rnn.PackedSequence`.

    To implement a model with recurrent layers, you can either use
    default container classes such as
    `pfrl.nn.RecurrentSequential` and
    `pfrl.nn.RecurrentBranched` or write your module
    extending this class and `torch.nn.Module`.
    """

    def forward(self, packed_input, recurrent_state):
        """Multi-step batch forward computation.

        Args:
            packed_input (object): Input sequences. Tensors must be packed in
                `torch.nn.utils.rnn.PackedSequence`.
            recurrent_state (object or None): Batched recurrent state.
                If set to None, it is initialized.

        Returns:
            object: Output sequences. Tensors will be packed in
                `torch.nn.utils.rnn.PackedSequence`.
            object or None: New batched recurrent state.
        """
        raise NotImplementedError
