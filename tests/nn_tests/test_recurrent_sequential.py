import unittest

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from pfrl.nn import Lambda, RecurrentSequential
from pfrl.testing import torch_assert_allclose


def _step_lstm(lstm, x, state):
    assert isinstance(lstm, nn.LSTM)
    lstm_cell = nn.LSTMCell(input_size=lstm.input_size, hidden_size=lstm.hidden_size)
    assert lstm.num_layers == 1
    lstm_cell.weight_ih = lstm.weight_ih_l0
    lstm_cell.weight_hh = lstm.weight_hh_l0
    lstm_cell.bias_ih = lstm.bias_ih_l0
    lstm_cell.bias_hh = lstm.bias_hh_l0
    h, c = lstm_cell(x, state)
    return h, (h, c)


def _step_rnn_tanh(rnn, x, state):
    assert isinstance(rnn, nn.RNN)
    rnn_cell = nn.RNNCell(input_size=rnn.input_size, hidden_size=rnn.hidden_size)
    rnn_cell.weight_ih = rnn.weight_ih_l0
    rnn_cell.weight_hh = rnn.weight_hh_l0
    rnn_cell.bias_ih = rnn.bias_ih_l0
    rnn_cell.bias_hh = rnn.bias_hh_l0
    h = rnn_cell(x, state)
    return h, h


class TestRecurrentSequential(unittest.TestCase):
    def _test_forward(self, gpu):
        in_size = 2
        out_size = 6

        rseq = RecurrentSequential(
            nn.Linear(in_size, 3),
            nn.ELU(),
            nn.LSTM(num_layers=1, input_size=3, hidden_size=4),
            nn.Linear(4, 5),
            nn.RNN(num_layers=1, input_size=5, hidden_size=out_size),
            nn.Tanh(),
        )

        if gpu >= 0:
            device = torch.device("cuda:{}".format(gpu))
            rseq.to(device)
        else:
            device = torch.device("cpu")

        assert len(rseq.recurrent_children) == 2
        assert rseq.recurrent_children[0] is rseq[2]
        assert rseq.recurrent_children[1] is rseq[4]

        linear1 = rseq[0]
        lstm = rseq[2]
        linear2 = rseq[3]
        rnn = rseq[4]

        seqs_x = [
            torch.rand(4, in_size, requires_grad=True, device=device),
            torch.rand(1, in_size, requires_grad=True, device=device),
            torch.rand(3, in_size, requires_grad=True, device=device),
        ]

        packed_x = nn.utils.rnn.pack_sequence(seqs_x, enforce_sorted=False)

        out, _ = rseq(packed_x, None)
        self.assertEqual(out.data.shape, (8, out_size))

        # Check if the output matches that of step-by-step execution
        def manual_forward(seqs_x):
            seqs_y = []
            for seq_x in seqs_x:
                lstm_st = None
                rnn_st = None
                seq_y = []
                for i in range(len(seq_x)):
                    h = seq_x[i : i + 1]
                    h = linear1(h)
                    h = F.elu(h)
                    h, lstm_st = _step_lstm(lstm, h, lstm_st)
                    h = linear2(h)
                    h, rnn_st = _step_rnn_tanh(rnn, h, rnn_st)
                    y = F.tanh(h)
                    seq_y.append(y[0])
                seqs_y.append(torch.stack(seq_y))
            return nn.utils.rnn.pack_sequence(seqs_y, enforce_sorted=False)

        manual_out = manual_forward(seqs_x)
        torch_assert_allclose(out.data, manual_out.data, atol=1e-4)

        # Finally, check the gradient (wrt input)
        grads = torch.autograd.grad([torch.sum(out.data)], seqs_x)
        manual_grads = torch.autograd.grad([torch.sum(manual_out.data)], seqs_x)
        assert len(grads) == len(manual_grads) == 3
        for grad, manual_grad in zip(grads, manual_grads):
            torch_assert_allclose(grad, manual_grad, atol=1e-4)

    @pytest.mark.gpu
    def test_forward_gpu(self):
        self._test_forward(gpu=0)

    def test_forward_cpu(self):
        self._test_forward(gpu=-1)

    def _test_forward_with_tuple_input(self, gpu):
        in_size = 5
        out_size = 3

        def concat_input(tensors):
            return torch.cat(tensors, dim=1)

        rseq = RecurrentSequential(
            Lambda(concat_input),
            nn.RNN(num_layers=1, input_size=in_size, hidden_size=out_size),
        )

        if gpu >= 0:
            device = torch.device("cuda:{}".format(gpu))
            rseq.to(device)
        else:
            device = torch.device("cpu")

        # Input is list of tuples. Each tuple has two variables.
        seqs_x = [
            (torch.rand(3, 2, device=device), torch.rand(3, 3, device=device)),
            (torch.rand(1, 2, device=device), torch.rand(1, 3, device=device)),
        ]
        packed_x = (
            nn.utils.rnn.pack_sequence([seqs_x[0][0], seqs_x[1][0]]),
            nn.utils.rnn.pack_sequence([seqs_x[0][1], seqs_x[1][1]]),
        )

        # Concatenated output should be a variable.
        out, _ = rseq(packed_x, None)
        self.assertEqual(out.data.shape, (4, out_size))

    @pytest.mark.gpu
    def test_forward_with_tuple_input_gpu(self):
        self._test_forward_with_tuple_input(gpu=0)

    def test_forward_with_tuple_input_cpu(self):
        self._test_forward_with_tuple_input(gpu=-1)

    def _test_forward_with_tuple_output(self, gpu):
        in_size = 5
        out_size = 6

        def split_output(x):
            return tuple(torch.split(x, [2, 1, 3], dim=1))

        rseq = RecurrentSequential(
            nn.RNN(num_layers=1, input_size=in_size, hidden_size=out_size),
            Lambda(split_output),
        )

        if gpu >= 0:
            device = torch.device("cuda:{}".format(gpu))
            rseq.to(device)
        else:
            device = torch.device("cpu")

        # Input is a list of two variables.
        seqs_x = [
            torch.rand(3, in_size, device=device),
            torch.rand(2, in_size, device=device),
        ]

        packed_x = nn.utils.rnn.pack_sequence(seqs_x, enforce_sorted=False)

        # Concatenated output should be a tuple of three variables.
        out, _ = rseq(packed_x, None)

        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0].data.shape, (5, 2))
        self.assertEqual(out[1].data.shape, (5, 1))
        self.assertEqual(out[2].data.shape, (5, 3))

    @pytest.mark.gpu
    def test_forward_with_tuple_output_gpu(self):
        self._test_forward_with_tuple_output(gpu=0)

    def test_forward_with_tuple_output_cpu(self):
        self._test_forward_with_tuple_output(gpu=-1)
