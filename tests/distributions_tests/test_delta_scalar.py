import pytest
import torch

from pfrl.distributions import Delta


@pytest.mark.parametrize("loc", [0.0, 2.5, -3, 0])
@pytest.mark.parametrize("validate_args", [None, False, True])
def test_delta_accepts_scalar_location(loc, validate_args):
    distribution = Delta(loc, validate_args=validate_args)
    sample = distribution.sample()

    assert distribution.batch_shape == torch.Size()
    torch.testing.assert_close(distribution.mean, torch.tensor(float(loc)))
    torch.testing.assert_close(distribution.variance, torch.tensor(0.0))
    torch.testing.assert_close(distribution.stddev, torch.tensor(0.0))
    torch.testing.assert_close(sample, torch.tensor(float(loc)))
    torch.testing.assert_close(distribution.rsample((2,)), torch.full((2,), float(loc)))
    expanded = distribution.expand((2, 3))
    torch.testing.assert_close(expanded.sample((4,)), torch.full((4, 2, 3), float(loc)))


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
)
def test_delta_keeps_tensor_location_and_gradients(device):
    if device != "cpu" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    loc = torch.tensor(
        [1.0, -2.0], dtype=torch.float64, device=device, requires_grad=True
    )
    distribution = Delta(loc)

    assert distribution.mean is loc
    assert distribution.batch_shape == torch.Size((2,))
    distribution.rsample((3,)).sum().backward()
    torch.testing.assert_close(
        loc.grad, torch.tensor([3.0, 3.0], dtype=torch.float64, device=device)
    )
