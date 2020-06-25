import pytest
import torch

import pfrl
from pfrl.testing import torch_assert_allclose


@pytest.mark.parametrize("n", [1, 5])
class TestConjugateGradient:
    @pytest.fixture(autouse=True)
    def setUp(self, n):
        self.n = n

    def _test(self, device):
        # A must be symmetric and positive-definite
        random_mat = torch.normal(0, 1, size=(self.n, self.n))
        random_mat.to(device)
        A = torch.matmul(random_mat, random_mat.T)
        x_ans = torch.normal(0, 1, size=(self.n,))
        x_ans.to(device)
        b = torch.matmul(A, x_ans)

        def A_product_func(vec):
            assert vec.shape == b.shape
            return torch.matmul(A, vec)

        x = pfrl.utils.conjugate_gradient(A_product_func, b)
        torch_assert_allclose(x, x_ans, rtol=1e-1)

    def test_cpu(self):
        self._test(torch.device("cpu"))

    @pytest.mark.gpu
    def test_gpu(self):
        self._test(torch.device("cuda:0"))
