import unittest

import numpy
import torch

from pfrl.nn import to_factorized_noisy


def names_of_parameters(module):
    return set([name for name, tensor in module.named_parameters()])


class TestToFactorizedNoisy(unittest.TestCase):
    def test_modulelist(self):
        model = torch.nn.ModuleList([torch.nn.Linear(1, 3), torch.nn.Linear(3, 4)])

        self.assertEqual(
            names_of_parameters(model), {"0.weight", "0.bias", "1.weight", "1.bias"}
        )
        to_factorized_noisy(model)
        self.assertEqual(
            names_of_parameters(model),
            {
                "0.mu.bias",
                "0.mu.weight",
                "0.sigma.bias",
                "0.sigma.weight",
                "1.mu.bias",
                "1.mu.weight",
                "1.sigma.bias",
                "1.sigma.weight",
            },
        )
        x = torch.as_tensor(numpy.ones((2, 1), numpy.float32))
        for layer in model:
            x = layer(x)
        x.sum().backward()
        for p in model.parameters():
            self.assertIsNotNone(p.grad)

    def test_sequential(self):
        model = torch.nn.Sequential(torch.nn.Linear(1, 3), torch.nn.Linear(3, 4))

        self.assertEqual(
            names_of_parameters(model), {"0.weight", "0.bias", "1.weight", "1.bias"}
        )
        to_factorized_noisy(model)
        self.assertEqual(
            names_of_parameters(model),
            {
                "0.mu.bias",
                "0.mu.weight",
                "0.sigma.bias",
                "0.sigma.weight",
                "1.mu.bias",
                "1.mu.weight",
                "1.sigma.bias",
                "1.sigma.weight",
            },
        )
        y = model(torch.as_tensor(numpy.ones((2, 1), numpy.float32)))
        y.sum().backward()
        for p in model.parameters():
            self.assertIsNotNone(p.grad)
