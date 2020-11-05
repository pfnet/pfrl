import unittest

import numpy as np
import torch
import torch.nn as nn

from pfrl.testing import torch_assert_allclose
from pfrl.utils import copy_param


class TestCopyParam(unittest.TestCase):
    def test_copy_param(self):
        a = nn.Linear(1, 5)
        b = nn.Linear(1, 5)

        s = torch.from_numpy(np.random.rand(1, 1).astype(np.float32))
        a_out = list(a(s).detach().numpy().ravel())
        b_out = list(b(s).detach().numpy().ravel())
        self.assertNotEqual(a_out, b_out)

        # Copy b's parameters to a
        copy_param.copy_param(a, b)

        a_out_new = list(a(s).detach().numpy().ravel())
        b_out_new = list(b(s).detach().numpy().ravel())
        self.assertEqual(a_out_new, b_out)
        self.assertEqual(b_out_new, b_out)

    def test_copy_param_scalar(self):
        a = nn.Module()
        a.p = nn.Parameter(torch.Tensor([1]))
        b = nn.Module()
        b.p = nn.Parameter(torch.Tensor([2]))

        self.assertNotEqual(a.p.detach().numpy(), b.p.detach().numpy())

        # Copy b's parameters to a
        copy_param.copy_param(a, b)

        self.assertEqual(a.p.detach().numpy(), b.p.detach().numpy())

    def test_copy_param_shape_check(self):
        a = nn.Linear(2, 5)
        b = nn.Linear(1, 5)

        with self.assertRaises(RuntimeError):
            # Different shape
            copy_param.copy_param(a, b)

        with self.assertRaises(RuntimeError):
            # Different shape
            copy_param.copy_param(b, a)

    def test_soft_copy_param(self):
        a = nn.Linear(1, 5)
        b = nn.Linear(1, 5)

        with torch.no_grad():
            a.weight.fill_(0.5)
            b.weight.fill_(1)

        # a = (1 - tau) * a + tau * b
        copy_param.soft_copy_param(target_link=a, source_link=b, tau=0.1)

        torch_assert_allclose(a.weight, torch.full_like(a.weight, 0.55))
        torch_assert_allclose(b.weight, torch.full_like(b.weight, 1.0))

        copy_param.soft_copy_param(target_link=a, source_link=b, tau=0.1)

        torch_assert_allclose(a.weight, torch.full_like(a.weight, 0.595))
        torch_assert_allclose(b.weight, torch.full_like(b.weight, 1.0))

    def test_soft_copy_param_scalar(self):
        a = nn.Module()
        a.p = nn.Parameter(torch.as_tensor(0.5))
        b = nn.Module()
        b.p = nn.Parameter(torch.as_tensor(1.0))

        # a = (1 - tau) * a + tau * b
        copy_param.soft_copy_param(target_link=a, source_link=b, tau=0.1)

        torch_assert_allclose(a.p, torch.full_like(a.p, 0.55))
        torch_assert_allclose(b.p, torch.full_like(b.p, 1.0))

        copy_param.soft_copy_param(target_link=a, source_link=b, tau=0.1)

        torch_assert_allclose(a.p, torch.full_like(a.p, 0.595))
        torch_assert_allclose(b.p, torch.full_like(b.p, 1.0))

    def test_soft_copy_param_shape_check(self):
        a = nn.Linear(2, 5)
        b = nn.Linear(1, 5)

        # Different shape
        with self.assertRaises(AssertionError):
            copy_param.soft_copy_param(a, b, 0.1)

        with self.assertRaises(AssertionError):
            copy_param.soft_copy_param(b, a, 0.1)

    def test_copy_grad(self):
        def set_random_grad(link):
            link.zero_grad()
            x = np.random.normal(size=(1, 1)).astype(np.float32)
            y = link(torch.from_numpy(x)) * np.random.normal()
            torch.sum(y).backward()

        # When source is not None and target is None
        a = nn.Linear(1, 5)
        b = nn.Linear(1, 5)
        set_random_grad(a)
        b.zero_grad()
        assert a.weight.grad is not None
        assert a.bias.grad is not None
        assert b.weight.grad is None
        assert b.bias.grad is None
        copy_param.copy_grad(target_link=b, source_link=a)
        torch_assert_allclose(a.weight.grad, b.weight.grad)
        torch_assert_allclose(a.bias.grad, b.bias.grad)
        assert a.weight.grad is not b.weight.grad
        assert a.bias.grad is not b.bias.grad

        # When both are not None
        a = nn.Linear(1, 5)
        b = nn.Linear(1, 5)
        set_random_grad(a)
        set_random_grad(b)
        assert a.weight.grad is not None
        assert a.bias.grad is not None
        assert b.weight.grad is not None
        assert b.bias.grad is not None
        copy_param.copy_grad(target_link=b, source_link=a)
        torch_assert_allclose(a.weight.grad, b.weight.grad)
        torch_assert_allclose(a.bias.grad, b.bias.grad)
        assert a.weight.grad is not b.weight.grad
        assert a.bias.grad is not b.bias.grad

        # When source is None and target is not None
        a = nn.Linear(1, 5)
        b = nn.Linear(1, 5)
        a.zero_grad()
        set_random_grad(b)
        assert a.weight.grad is None
        assert a.bias.grad is None
        assert b.weight.grad is not None
        assert b.bias.grad is not None
        copy_param.copy_grad(target_link=b, source_link=a)
        assert a.weight.grad is None
        assert a.bias.grad is None
        assert b.weight.grad is None
        assert b.bias.grad is None

        # When both are None
        a = nn.Linear(1, 5)
        b = nn.Linear(1, 5)
        a.zero_grad()
        b.zero_grad()
        assert a.weight.grad is None
        assert a.bias.grad is None
        assert b.weight.grad is None
        assert b.bias.grad is None
        copy_param.copy_grad(target_link=b, source_link=a)
        assert a.weight.grad is None
        assert a.bias.grad is None
        assert b.weight.grad is None
        assert b.bias.grad is None
