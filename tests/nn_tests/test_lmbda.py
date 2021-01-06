import torch
from torch import nn

from pfrl.nn.lmbda import Lambda
from pfrl.testing import torch_assert_allclose


def test_lambda():
    model = nn.Sequential(
        nn.ReLU(),
        Lambda(lambda x: x + 1),
        nn.ReLU(),
    )
    x = torch.rand(3, 2)
    # Since x is all positive, ReLU will not have any effects
    y = model(x)
    torch_assert_allclose(y, x + 1)
