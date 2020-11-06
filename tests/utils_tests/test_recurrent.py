import unittest

import pytest
import torch
from torch import nn

from pfrl.testing import torch_assert_allclose
from pfrl.utils.recurrent import (
    concatenate_recurrent_states,
    get_recurrent_state_at,
    mask_recurrent_state_at,
)


class TestRecurrentStateFunctions(unittest.TestCase):
    def _test_lstm(self, gpu):
        in_size = 2
        out_size = 3
        device = "cuda:{}".format(gpu) if gpu >= 0 else "cpu"
        seqs_x = [
            torch.rand(4, in_size, device=device),
            torch.rand(1, in_size, device=device),
            torch.rand(3, in_size, device=device),
        ]
        seqs_x = torch.nn.utils.rnn.pack_sequence(seqs_x, enforce_sorted=False)
        link = nn.LSTM(num_layers=1, input_size=in_size, hidden_size=out_size)
        link.to(device)

        # Forward twice: with None and non-None random states
        y0, rs0 = link(seqs_x, None)
        y1, rs1 = link(seqs_x, rs0)
        y0, _ = torch.nn.utils.rnn.pad_packed_sequence(y0, batch_first=True)
        y1, _ = torch.nn.utils.rnn.pad_packed_sequence(y1, batch_first=True)
        h0, c0 = rs0
        h1, c1 = rs1
        self.assertEqual(y0.shape, (3, 4, out_size))
        self.assertEqual(y1.shape, (3, 4, out_size))
        self.assertEqual(c0.shape, (1, 3, out_size))
        self.assertEqual(h0.shape, (1, 3, out_size))
        self.assertEqual(c0.shape, (1, 3, out_size))
        self.assertEqual(h1.shape, (1, 3, out_size))
        self.assertEqual(c1.shape, (1, 3, out_size))

        # Masked at 0
        rs0_mask0 = mask_recurrent_state_at(rs0, 0)
        y1m0, _ = link(seqs_x, rs0_mask0)
        y1m0, _ = torch.nn.utils.rnn.pad_packed_sequence(y1m0, batch_first=True)
        torch_assert_allclose(y1m0[0], y0[0])
        torch_assert_allclose(y1m0[1], y1[1])
        torch_assert_allclose(y1m0[2], y1[2])

        # Masked at 1
        rs0_mask1 = mask_recurrent_state_at(rs0, 1)
        y1m1, _ = link(seqs_x, rs0_mask1)
        y1m1, _ = torch.nn.utils.rnn.pad_packed_sequence(y1m1, batch_first=True)
        torch_assert_allclose(y1m1[0], y1[0])
        torch_assert_allclose(y1m1[1], y0[1])
        torch_assert_allclose(y1m1[2], y1[2])

        # Masked at (1, 2)
        rs0_mask12 = mask_recurrent_state_at(rs0, (1, 2))
        y1m12, _ = link(seqs_x, rs0_mask12)
        y1m12, _ = torch.nn.utils.rnn.pad_packed_sequence(y1m12, batch_first=True)
        torch_assert_allclose(y1m12[0], y1[0])
        torch_assert_allclose(y1m12[1], y0[1])
        torch_assert_allclose(y1m12[2], y0[2])

        # Get at 1 and concat with None
        rs0_get1 = get_recurrent_state_at(rs0, 1, detach=False)
        assert rs0_get1[0].requires_grad
        assert rs0_get1[1].requires_grad
        torch_assert_allclose(rs0_get1[0], h0[:, 1])
        torch_assert_allclose(rs0_get1[1], c0[:, 1])
        concat_rs_get1 = concatenate_recurrent_states([None, rs0_get1, None])
        y1g1, _ = link(seqs_x, concat_rs_get1)
        y1g1, _ = torch.nn.utils.rnn.pad_packed_sequence(y1g1, batch_first=True)
        torch_assert_allclose(y1g1[0], y0[0])
        torch_assert_allclose(y1g1[1], y1[1])
        torch_assert_allclose(y1g1[2], y0[2])

        # Get at 1 with detach=True
        rs0_get1_detach = get_recurrent_state_at(rs0, 1, detach=True)
        assert not rs0_get1_detach[0].requires_grad
        assert not rs0_get1_detach[1].requires_grad
        torch_assert_allclose(rs0_get1_detach[0], h0[:, 1])
        torch_assert_allclose(rs0_get1_detach[1], c0[:, 1])

    @pytest.mark.gpu
    def test_lstm_gpu(self):
        self._test_lstm(gpu=0)

    def test_lstm_cpu(self):
        self._test_lstm(gpu=-1)

    def _test_non_lstm(self, gpu, name):
        in_size = 2
        out_size = 3
        device = "cuda:{}".format(gpu) if gpu >= 0 else "cpu"
        seqs_x = [
            torch.rand(4, in_size, device=device),
            torch.rand(1, in_size, device=device),
            torch.rand(3, in_size, device=device),
        ]
        seqs_x = torch.nn.utils.rnn.pack_sequence(seqs_x, enforce_sorted=False)
        self.assertTrue(name in ("GRU", "RNN"))
        cls = getattr(nn, name)
        link = cls(num_layers=1, input_size=in_size, hidden_size=out_size)
        link.to(device)

        # Forward twice: with None and non-None random states
        y0, h0 = link(seqs_x, None)
        y1, h1 = link(seqs_x, h0)
        y0, _ = torch.nn.utils.rnn.pad_packed_sequence(y0, batch_first=True)
        y1, _ = torch.nn.utils.rnn.pad_packed_sequence(y1, batch_first=True)
        self.assertEqual(h0.shape, (1, 3, out_size))
        self.assertEqual(h1.shape, (1, 3, out_size))
        self.assertEqual(y0.shape, (3, 4, out_size))
        self.assertEqual(y1.shape, (3, 4, out_size))

        # Masked at 0
        rs0_mask0 = mask_recurrent_state_at(h0, 0)
        y1m0, _ = link(seqs_x, rs0_mask0)
        y1m0, _ = torch.nn.utils.rnn.pad_packed_sequence(y1m0, batch_first=True)
        torch_assert_allclose(y1m0[0], y0[0])
        torch_assert_allclose(y1m0[1], y1[1])
        torch_assert_allclose(y1m0[2], y1[2])

        # Masked at (1, 2)
        rs0_mask12 = mask_recurrent_state_at(h0, (1, 2))
        y1m12, _ = link(seqs_x, rs0_mask12)
        y1m12, _ = torch.nn.utils.rnn.pad_packed_sequence(y1m12, batch_first=True)
        torch_assert_allclose(y1m12[0], y1[0])
        torch_assert_allclose(y1m12[1], y0[1])
        torch_assert_allclose(y1m12[2], y0[2])

        # Get at 1 and concat with None
        rs0_get1 = get_recurrent_state_at(h0, 1, detach=False)
        assert rs0_get1.requires_grad
        torch_assert_allclose(rs0_get1, h0[:, 1])
        concat_rs_get1 = concatenate_recurrent_states([None, rs0_get1, None])
        y1g1, _ = link(seqs_x, concat_rs_get1)
        y1g1, _ = torch.nn.utils.rnn.pad_packed_sequence(y1g1, batch_first=True)
        torch_assert_allclose(y1g1[0], y0[0])
        torch_assert_allclose(y1g1[1], y1[1])
        torch_assert_allclose(y1g1[2], y0[2])

        # Get at 1 with detach=True
        rs0_get1_detach = get_recurrent_state_at(h0, 1, detach=True)
        assert not rs0_get1_detach.requires_grad
        torch_assert_allclose(rs0_get1_detach, h0[:, 1])

    @pytest.mark.gpu
    def test_gru_gpu(self):
        self._test_non_lstm(gpu=0, name="GRU")

    def test_gru_cpu(self):
        self._test_non_lstm(gpu=-1, name="GRU")

    @pytest.mark.gpu
    def test_rnn_gpu(self):
        self._test_non_lstm(gpu=0, name="RNN")

    def test_rnn_cpu(self):
        self._test_non_lstm(gpu=-1, name="RNN")
