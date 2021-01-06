import pytest
import torch
from torch import nn

from pfrl.nn import Branched
from pfrl.testing import torch_assert_allclose


@pytest.mark.parametrize("batch_size", [1, 2])
def test_branched(batch_size):
    link1 = nn.Linear(2, 3)
    link2 = nn.Linear(2, 5)
    link3 = nn.Sequential(
        nn.Linear(2, 7),
        nn.Tanh(),
    )
    plink = Branched(link1, link2, link3)
    x = torch.zeros(batch_size, 2, dtype=torch.float)
    pout = plink(x)
    assert isinstance(pout, tuple)
    assert len(pout) == 3
    out1 = link1(x)
    out2 = link2(x)
    out3 = link3(x)
    torch_assert_allclose(pout[0], out1)
    torch_assert_allclose(pout[1], out2)
    torch_assert_allclose(pout[2], out3)
