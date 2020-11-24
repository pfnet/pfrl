import random
import unittest

import pytest
import torch

import pfrl


class TestSetRandomSeed(unittest.TestCase):
    def test_random(self):
        pfrl.utils.set_random_seed(0)
        seed0_0 = random.random()
        pfrl.utils.set_random_seed(1)
        seed1_0 = random.random()
        pfrl.utils.set_random_seed(0)
        seed0_1 = random.random()
        pfrl.utils.set_random_seed(1)
        seed1_1 = random.random()
        self.assertEqual(seed0_0, seed0_1)
        self.assertEqual(seed1_0, seed1_1)
        self.assertNotEqual(seed0_0, seed1_0)

    def _test_random_device(self, device):
        pfrl.utils.set_random_seed(0)
        seed0_0 = torch.rand(1, device=device)
        pfrl.utils.set_random_seed(1)
        seed1_0 = torch.rand(1, device=device)
        pfrl.utils.set_random_seed(0)
        seed0_1 = torch.rand(1, device=device)
        pfrl.utils.set_random_seed(1)
        seed1_1 = torch.rand(1, device=device)
        self.assertEqual(seed0_0, seed0_1)
        self.assertEqual(seed1_0, seed1_1)
        self.assertNotEqual(seed0_0, seed1_0)

    def test_random_cpu(self):
        device = torch.device("cpu")
        self._test_random_device(device)

    @pytest.mark.gpu
    def test_random_gpu(self):
        device = torch.device("cuda:0")
        self.assertTrue(torch.cuda.is_available())
        self._test_random_device(device)
