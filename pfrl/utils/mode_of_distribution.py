import torch
from torch import distributions as dists


def mode_of_distribution(distrib):
    assert isinstance(distrib, dists.Distribution)
    if isinstance(distrib, dists.Categorical):
        return distrib.probs.argmax(dim=-1)
    elif isinstance(distrib, (dists.Normal, dists.MultivariateNormal)):
        return distrib.mean
    elif isinstance(distrib, dists.transformed_distribution.TransformedDistribution):
        x = mode_of_distribution(distrib.base_dist)
        for transform in distrib.transforms:
            x = transform(x)
        return x
    elif isinstance(distrib, torch.distributions.Independent):
        return mode_of_distribution(distrib.base_dist)
    else:
        raise RuntimeError("{} is not supported".format(distrib))
