import unittest

import numpy as np
import pytest
import torch

import pfrl


class TestBatchStates(unittest.TestCase):
    def _test(self, gpu):

        # state: ((2,2)-shaped array, integer, (1,)-shaped array)
        states = [
            (np.arange(4).reshape((2, 2)), 0, np.zeros(1)),
            (np.arange(4).reshape((2, 2)) + 1, 1, np.zeros(1) + 1),
        ]
        if gpu >= 0:
            device = torch.device("cuda:{}".format(gpu))
        else:
            device = torch.device("cpu")

        def phi(state):
            return state[0] * 2, state[1], state[2] * 3

        batch = pfrl.utils.batch_states(states, device=device, phi=phi)
        self.assertIsInstance(batch, tuple)
        batch_a, batch_b, batch_c = batch
        np.testing.assert_allclose(
            batch_a.cpu(),
            np.asarray(
                [
                    [[0, 2], [4, 6]],
                    [[2, 4], [6, 8]],
                ]
            ),
        )
        np.testing.assert_allclose(batch_b.cpu(), np.asarray([0, 1]))
        np.testing.assert_allclose(
            batch_c.cpu(),
            np.asarray(
                [
                    [0],
                    [3],
                ]
            ),
        )

    def test_cpu(self):
        self._test(gpu=-1)

    @pytest.mark.gpu
    def test_gpu(self):
        self._test(gpu=0)
