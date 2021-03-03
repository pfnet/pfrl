import unittest

import pytest
import torch
from torch import nn

from pfrl.nn import RecurrentBranched, RecurrentSequential
from pfrl.testing import torch_assert_allclose
from pfrl.utils.recurrent import (
    concatenate_recurrent_states,
    get_recurrent_state_at,
    mask_recurrent_state_at,
    one_step_forward,
)


class TestRecurrentBranched(unittest.TestCase):
    def _test_forward(self, gpu):
        in_size = 2
        out0_size = 3
        out1_size = 4
        out2_size = 1

        par = RecurrentBranched(
            nn.LSTM(num_layers=1, input_size=in_size, hidden_size=out0_size),
            RecurrentSequential(
                nn.RNN(num_layers=1, input_size=in_size, hidden_size=out1_size),
            ),
            RecurrentSequential(
                nn.Linear(in_size, out2_size),
            ),
        )

        if gpu >= 0:
            device = torch.device("cuda:{}".format(gpu))
            par.to(device)
        else:
            device = torch.device("cpu")

        seqs_x = [
            torch.rand(1, in_size, device=device),
            torch.rand(3, in_size, device=device),
        ]

        packed_x = nn.utils.rnn.pack_sequence(seqs_x, enforce_sorted=False)

        # Concatenated output should be a tuple of three variables.
        out, rs = par(packed_x, None)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), len(par))
        self.assertEqual(out[0].data.shape, (4, out0_size))
        self.assertEqual(out[1].data.shape, (4, out1_size))
        self.assertEqual(out[2].data.shape, (4, out2_size))

        self.assertIsInstance(rs, tuple)
        self.assertEqual(len(rs), len(par))

        # LSTM
        self.assertIsInstance(rs[0], tuple)
        self.assertEqual(len(rs[0]), 2)
        self.assertEqual(rs[0][0].shape, (1, len(seqs_x), out0_size))
        self.assertEqual(rs[0][1].shape, (1, len(seqs_x), out0_size))

        # RecurrentSequential(RNN)
        self.assertIsInstance(rs[1], tuple)
        self.assertEqual(len(rs[1]), 1)
        self.assertEqual(rs[1][0].shape, (1, len(seqs_x), out1_size))

        # RecurrentSequential(Linear)
        self.assertIsInstance(rs[2], tuple)
        self.assertEqual(len(rs[2]), 0)

    @pytest.mark.gpu
    def test_forward_gpu(self):
        self._test_forward(gpu=0)

    def test_forward_cpu(self):
        self._test_forward(gpu=-1)

    def _test_forward_with_modified_recurrent_state(self, gpu):
        in_size = 2
        out0_size = 2
        out1_size = 3
        par = RecurrentBranched(
            nn.GRU(num_layers=1, input_size=in_size, hidden_size=out0_size),
            RecurrentSequential(
                nn.LSTM(num_layers=1, input_size=in_size, hidden_size=out1_size),
            ),
        )
        if gpu >= 0:
            device = torch.device("cuda:{}".format(gpu))
            par.to(device)
        else:
            device = torch.device("cpu")
        seqs_x = [
            torch.rand(2, in_size, device=device),
            torch.rand(2, in_size, device=device),
        ]
        packed_x = nn.utils.rnn.pack_sequence(seqs_x, enforce_sorted=False)
        x_t0 = torch.stack((seqs_x[0][0], seqs_x[1][0]))
        x_t1 = torch.stack((seqs_x[0][1], seqs_x[1][1]))

        (gru_out, lstm_out), (gru_rs, (lstm_rs,)) = par(packed_x, None)

        # Check if n_step_forward and forward twice results are same
        def no_mask_forward_twice():
            _, rs = one_step_forward(par, x_t0, None)
            return one_step_forward(par, x_t1, rs)

        (
            (nomask_gru_out, nomask_lstm_out),
            (nomask_gru_rs, (nomask_lstm_rs,)),
        ) = no_mask_forward_twice()

        # GRU
        torch_assert_allclose(gru_out.data[2:], nomask_gru_out, atol=1e-5)
        torch_assert_allclose(gru_rs, nomask_gru_rs)

        # LSTM
        torch_assert_allclose(lstm_out.data[2:], nomask_lstm_out, atol=1e-5)
        torch_assert_allclose(lstm_rs[0], nomask_lstm_rs[0], atol=1e-5)
        torch_assert_allclose(lstm_rs[1], nomask_lstm_rs[1], atol=1e-5)

        # 1st-only mask forward twice: only 2nd should be the same
        def mask0_forward_twice():
            _, rs = one_step_forward(par, x_t0, None)
            rs = mask_recurrent_state_at(rs, 0)
            return one_step_forward(par, x_t1, rs)

        (
            (mask0_gru_out, mask0_lstm_out),
            (mask0_gru_rs, (mask0_lstm_rs,)),
        ) = mask0_forward_twice()

        # GRU
        with self.assertRaises(AssertionError):
            torch_assert_allclose(gru_out.data[2], mask0_gru_out[0], atol=1e-5)
        torch_assert_allclose(gru_out.data[3], mask0_gru_out[1], atol=1e-5)

        # LSTM
        with self.assertRaises(AssertionError):
            torch_assert_allclose(lstm_out.data[2], mask0_lstm_out[0], atol=1e-5)
        torch_assert_allclose(lstm_out.data[3], mask0_lstm_out[1], atol=1e-5)

        # 2nd-only mask forward twice: only 1st should be the same
        def mask1_forward_twice():
            _, rs = one_step_forward(par, x_t0, None)
            rs = mask_recurrent_state_at(rs, 1)
            return one_step_forward(par, x_t1, rs)

        (
            (mask1_gru_out, mask1_lstm_out),
            (mask1_gru_rs, (mask1_lstm_rs,)),
        ) = mask1_forward_twice()

        # GRU
        torch_assert_allclose(gru_out.data[2], mask1_gru_out[0], atol=1e-5)
        with self.assertRaises(AssertionError):
            torch_assert_allclose(gru_out.data[3], mask1_gru_out[1], atol=1e-5)

        # LSTM
        torch_assert_allclose(lstm_out.data[2], mask1_lstm_out[0], atol=1e-5)
        with self.assertRaises(AssertionError):
            torch_assert_allclose(lstm_out.data[3], mask1_lstm_out[1], atol=1e-5)

        # both 1st and 2nd mask forward twice: both should be different
        def mask01_forward_twice():
            _, rs = one_step_forward(par, x_t0, None)
            rs = mask_recurrent_state_at(rs, [0, 1])
            return one_step_forward(par, x_t1, rs)

        (
            (mask01_gru_out, mask01_lstm_out),
            (mask01_gru_rs, (mask01_lstm_rs,)),
        ) = mask01_forward_twice()

        # GRU
        with self.assertRaises(AssertionError):
            torch_assert_allclose(gru_out.data[2], mask01_gru_out[0], atol=1e-5)
        with self.assertRaises(AssertionError):
            torch_assert_allclose(gru_out.data[3], mask01_gru_out[1], atol=1e-5)

        # LSTM
        with self.assertRaises(AssertionError):
            torch_assert_allclose(lstm_out.data[2], mask01_lstm_out[0], atol=1e-5)
        with self.assertRaises(AssertionError):
            torch_assert_allclose(lstm_out.data[3], mask01_lstm_out[1], atol=1e-5)

        # get and concat recurrent states and resume forward
        def get_and_concat_rs_forward():
            _, rs = one_step_forward(par, x_t0, None)
            rs0 = get_recurrent_state_at(rs, 0, detach=True)
            rs1 = get_recurrent_state_at(rs, 1, detach=True)
            concat_rs = concatenate_recurrent_states([rs0, rs1])
            return one_step_forward(par, x_t1, concat_rs)

        (
            (getcon_gru_out, getcon_lstm_out),
            (getcon_gru_rs, (getcon_lstm_rs,)),
        ) = get_and_concat_rs_forward()

        # GRU
        torch_assert_allclose(gru_out.data[2], getcon_gru_out[0], atol=1e-5)
        torch_assert_allclose(gru_out.data[3], getcon_gru_out[1], atol=1e-5)

        # LSTM
        torch_assert_allclose(lstm_out.data[2], getcon_lstm_out[0], atol=1e-5)
        torch_assert_allclose(lstm_out.data[3], getcon_lstm_out[1], atol=1e-5)

    @pytest.mark.gpu
    def test_forward_with_modified_recurrent_state_gpu(self):
        self._test_forward_with_modified_recurrent_state(gpu=0)

    def test_forward_with_modified_recurrent_state_cpu(self):
        self._test_forward_with_modified_recurrent_state(gpu=-1)
