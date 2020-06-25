import pytest
import torch

from pfrl.testing import torch_assert_allclose


def test_torch_assert_allclose():
    x = [torch.zeros(2), torch.ones(2)]
    y = [[0, 0], [1, 1]]
    torch_assert_allclose(x, y)


def test_torch_assert_allclose_fail():
    with pytest.raises(AssertionError):
        x = [torch.zeros(2), torch.ones(2)]
        y = [[0, 0], [1, 0]]
        torch_assert_allclose(x, y)
