import math

import torch

from pfrl.testing import torch_assert_allclose
from pfrl.utils.mode_of_distribution import mode_of_distribution


def test_transform():
    base_dist = torch.distributions.Normal(loc=2, scale=1)
    dist = torch.distributions.TransformedDistribution(
        base_dist, [torch.distributions.transforms.TanhTransform()]
    )
    mode = mode_of_distribution(dist)
    torch_assert_allclose(mode.tolist(), math.tanh(2))


def test_categorical():
    probs = torch.as_tensor([0.2, 0.6, 0.2])
    dist = torch.distributions.Categorical(probs=probs)
    mode = mode_of_distribution(dist)
    assert mode.tolist() == 1

    probs = torch.as_tensor([0.6, 0.2, 0.2])
    dist = torch.distributions.Categorical(probs=probs)
    mode = mode_of_distribution(dist)
    assert mode.tolist() == 0

    probs = torch.as_tensor([[0.6, 0.2, 0.2], [0.2, 0.2, 0.6]])
    dist = torch.distributions.Categorical(probs)
    mode = mode_of_distribution(dist)
    assert mode.tolist() == [0, 2]


def test_normal():
    loc = torch.as_tensor([0.3, 0.5])
    scale = torch.as_tensor([0.1, 0.9])
    dist = torch.distributions.Normal(loc, scale)
    mode = mode_of_distribution(dist)
    torch_assert_allclose(mode, loc)


def test_multivariate_normal():
    loc = torch.as_tensor([0.3, 0.7])
    cov = torch.as_tensor([[0.1, 0.0], [0.0, 0.9]])
    dist = torch.distributions.MultivariateNormal(loc, cov)
    mode = mode_of_distribution(dist)
    torch_assert_allclose(mode, loc)


def test_independent_normal():
    loc = torch.as_tensor([[0.3, 0.7], [0.2, 0.4]])
    scale = torch.as_tensor([[0.1, 0.2], [0.3, 0.8]])
    dist = torch.distributions.Independent(torch.distributions.Normal(loc, scale), 1)
    mode = mode_of_distribution(dist)
    torch_assert_allclose(mode, loc)
