import numpy
import pytest
import torch

from pfrl.nn import noisy_linear


@pytest.mark.parametrize("bias", [False, True])
class TestFactorizedNoisyLinear:
    @pytest.fixture(autouse=True)
    def setUp(self, bias):
        self.bias = bias
        mu = torch.nn.Linear(6, 5, bias=self.bias)
        self.linear = noisy_linear.FactorizedNoisyLinear(mu)

    def _test_calls(self, device):
        x_data = torch.arange(12, device=device, dtype=torch.float32).reshape((2, 6))
        x = torch.as_tensor(x_data, device=device)
        self.linear(x)
        self.linear(x_data + 1)

    def test_calls_cpu(self):
        self._test_calls(device=torch.device("cpu"))

    @pytest.mark.gpu
    def test_calls_gpu(self):
        device = torch.device("cuda")
        self.linear.to(device)
        self._test_calls(device)

    @pytest.mark.gpu
    def test_calls_gpu_after_to_gpu(self):
        device = torch.device("cuda")
        mu = self.linear.mu
        mu.to(device)
        self.linear = noisy_linear.FactorizedNoisyLinear(mu)
        self._test_calls(device)

    def _test_randomness(self, device):
        x = torch.normal(0, 1, size=(10, 6), device=device).float()
        y1 = self.linear(x).detach().cpu().numpy()
        y2 = self.linear(x).detach().cpu().numpy()
        d = float(numpy.mean(numpy.square(y1 - y2)))

        # The parameter name suggests that
        # torch.sqrt(d / 2) is approx to sigma_scale = 0.4
        # In fact, (for each element _[i, j],) it holds:
        # \E[(y2 - y1) ** 2] = 2 * \Var(y) = (4 / pi) * sigma_scale ** 2

        target = (0.4 ** 2) * 2
        if self.bias:
            target *= 2 / numpy.pi + numpy.sqrt(2 / numpy.pi) / y1.shape[1]
        else:
            target *= 2 / numpy.pi

        assert d > target / 3.0
        assert d < target * 3.0

    def test_randomness_cpu(self):
        device = torch.device("cpu")
        self._test_randomness(device)

    @pytest.mark.gpu
    def test_randomness_gpu(self):
        device = torch.device("cuda")
        self.linear.to(device)
        self._test_randomness(device)

    def _test_non_randomness(self, device):
        # Noises should be the same in a batch
        x0 = torch.normal(0, 1, size=(1, 6), dtype=torch.float32, device=device)
        x = x0.repeat(2, 1)
        y = self.linear(x)
        torch.testing.assert_allclose(y[0], y[1], rtol=1e-4, atol=0)

    def test_non_randomness_cpu(self):
        self._test_non_randomness(torch.device("cpu"))

    @pytest.mark.gpu
    def test_non_randomness_gpu(self):
        device = torch.device("cuda")
        self.linear.to(device)
        self._test_non_randomness(device)
