import torch
from torch import distributions

from pfrl.utils.sample_with_log_prob import sample_with_log_prob


def test_normal():
    distrib = distributions.Normal(loc=2, scale=1)
    for _ in range(10):
        y, log_prob = sample_with_log_prob(distrib)
        log_p = distrib.log_prob(y)
        torch.allclose(log_p, log_prob)


def test_transform():
    distrib = distributions.TransformedDistribution(
        distributions.Normal(loc=2, scale=1), [distributions.transforms.TanhTransform()]
    )
    for _ in range(10):
        y, log_prob = sample_with_log_prob(distrib)
        log_p = distrib.log_prob(y)
        torch.allclose(log_p, log_prob)


def test_transform_mv_normal():
    loc = torch.as_tensor([0.3, 0.7])
    cov = torch.as_tensor([[0.1, 0.0], [0.0, 0.9]])
    base_dist = distributions.MultivariateNormal(loc, cov)
    distrib = distributions.TransformedDistribution(
        base_dist, [distributions.transforms.TanhTransform()]
    )
    for _ in range(10):
        y, log_prob = sample_with_log_prob(distrib)
        log_p = distrib.log_prob(y)
        torch.allclose(log_p, log_prob)


def test_transform_independent_normal():
    # batch size is 3
    loc = torch.as_tensor([[0.3, 0.7], [0.2, 0.4], [0.8, 0.9]])
    scale = torch.as_tensor([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]])
    base_dist = distributions.Independent(distributions.Normal(loc, scale), 1)
    distrib = distributions.TransformedDistribution(
        base_dist, [distributions.transforms.TanhTransform()]
    )
    for _ in range(10):
        y, log_prob = sample_with_log_prob(distrib)
        log_p = distrib.log_prob(y)
        torch.allclose(log_p, log_prob)


def test_transform_batch_mv_normal():
    # batch size is 3
    loc = torch.as_tensor([[0.3, 0.7], [0.2, 0.4], [0.8, 0.9]])
    cov = torch.as_tensor(
        [[[0.1, 0.0], [0.0, 0.9]], [[0.5, 0.0], [0.0, 0.6]], [[0.7, 0.0], [0.0, 0.1]]]
    )
    base_dist = distributions.MultivariateNormal(loc, cov)
    distrib = distributions.TransformedDistribution(
        base_dist, [distributions.transforms.TanhTransform()]
    )
    for _ in range(10):
        y, log_prob = sample_with_log_prob(distrib)
        log_p = distrib.log_prob(y)
        torch.allclose(log_p, log_prob)


def test_multivariate_normal():
    loc = torch.as_tensor([0.3, 0.7])
    cov = torch.as_tensor([[0.1, 0.0], [0.0, 0.9]])
    dist = distributions.MultivariateNormal(loc, cov)
    for _ in range(10):
        y, log_prob = sample_with_log_prob(dist)
        log_p = dist.log_prob(y)
        torch.allclose(log_p, log_prob)
