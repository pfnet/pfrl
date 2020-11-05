import unittest

import numpy as np
import pytest
import torch

import pfrl

assertions = unittest.TestCase("__init__")


@pytest.mark.parametrize("in_size", [1, 5])
@pytest.mark.parametrize("out_size", [1, 3])
@pytest.mark.parametrize("hidden_sizes", [(), (1,), (1, 1), (7, 8)])
@pytest.mark.parametrize("normalize_input", [True, False])
@pytest.mark.parametrize("normalize_output", [True, False])
@pytest.mark.parametrize("nonlinearity", ["relu", "elu"])
@pytest.mark.parametrize("last_wscale", [1, 1e-3])
class TestMLPBN:
    @pytest.fixture(autouse=True)
    def setUp(
        self,
        in_size,
        out_size,
        hidden_sizes,
        normalize_input,
        normalize_output,
        nonlinearity,
        last_wscale,
    ):
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_sizes = hidden_sizes
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output
        self.nonlinearity = nonlinearity
        self.last_wscale = last_wscale

    def _test_call(self, gpu):
        nonlinearity = getattr(torch.nn.functional, self.nonlinearity)
        mlp = pfrl.nn.MLPBN(
            in_size=self.in_size,
            out_size=self.out_size,
            hidden_sizes=self.hidden_sizes,
            normalize_input=self.normalize_input,
            normalize_output=self.normalize_output,
            nonlinearity=nonlinearity,
            last_wscale=self.last_wscale,
        )
        batch_size = 7
        x = np.random.rand(batch_size, self.in_size).astype(np.float32)
        x = torch.from_numpy(x)
        if gpu >= 0:
            assertions.assertTrue(torch.cuda.is_available())
            device = torch.device("cuda:{}".format(gpu))
            mlp = mlp.to(device)
            x = x.cuda()
        y = mlp(x)
        assertions.assertEqual(y.shape, (batch_size, self.out_size))
        assertions.assertEqual(y.device, x.device)

    def test_call_cpu(self):
        self._test_call(gpu=-1)

    @pytest.mark.gpu
    def test_call_gpu(self):
        self._test_call(gpu=0)
