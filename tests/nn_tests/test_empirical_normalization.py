import unittest

import numpy as np
import pytest
import torch

from pfrl.nn import empirical_normalization


class TestEmpiricalNormalization(unittest.TestCase):
    def test_small_cpu(self):
        self._test_small(gpu=-1)

    @pytest.mark.gpu
    def test_small_gpu(self):
        self._test_small(gpu=0)

    def _test_small(self, gpu):
        en = empirical_normalization.EmpiricalNormalization(10)
        if gpu >= 0:
            device = "cuda:{}".format(gpu)
            en.to(device)
        else:
            device = "cpu"

        xs = []
        for t in range(10):
            x = np.random.normal(loc=4, scale=2, size=(t + 3, 10))
            en(torch.tensor(x, device=device))
            xs.extend(list(x))
        xs = np.stack(xs)
        true_mean = np.mean(xs, axis=0)
        true_std = np.std(xs, axis=0)
        np.testing.assert_allclose(en.mean.cpu().numpy(), true_mean, rtol=1e-4)
        np.testing.assert_allclose(en.std.cpu().numpy(), true_std, rtol=1e-4)

    @pytest.mark.slow
    def test_large(self):
        en = empirical_normalization.EmpiricalNormalization(10)
        for _ in range(10000):
            x = np.random.normal(loc=4, scale=2, size=(7, 10))
            en(torch.tensor(x))
        x = 2 * np.random.normal(loc=4, scale=2, size=(1, 10))
        enx = en(torch.tensor(x), update=False)

        np.testing.assert_allclose(en.mean.cpu().numpy(), 4, rtol=1e-1)
        np.testing.assert_allclose(en.std.cpu().numpy(), 2, rtol=1e-1)

        # Compare with the ground-truth normalization
        np.testing.assert_allclose((x - 4) / 2, enx, rtol=1e-1)

        # Test inverse
        np.testing.assert_allclose(x, en.inverse(torch.tensor(enx)), rtol=1e-4)

    def test_batch_axis(self):
        shape = (2, 3, 4)
        for batch_axis in range(3):
            en = empirical_normalization.EmpiricalNormalization(
                shape=shape[:batch_axis] + shape[batch_axis + 1 :],
                batch_axis=batch_axis,
            )
            for _ in range(10):
                x = np.random.rand(*shape)
                en(torch.tensor(x))

    def test_until(self):
        en = empirical_normalization.EmpiricalNormalization(7, until=20)
        last_mean = None
        last_std = None
        for t in range(15):
            en(torch.tensor(np.random.rand(2, 7) + t))

            if 1 <= t < 10:
                self.assertFalse(
                    np.allclose(en.mean.cpu().numpy(), last_mean, rtol=1e-4)
                )
                self.assertFalse(np.allclose(en.std.cpu().numpy(), last_std, rtol=1e-4))
            elif t >= 10:
                np.testing.assert_allclose(en.mean.cpu().numpy(), last_mean, rtol=1e-4)
                np.testing.assert_allclose(en.std.cpu().numpy(), last_std, rtol=1e-4)

            last_mean = en.mean
            last_std = en.std
