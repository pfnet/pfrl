import numpy as np
import pytest
import torch

from pfrl.functions.lower_triangular_matrix import lower_triangular_matrix


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
class TestLowerTriangularMatrix:
    @pytest.fixture(autouse=True)
    def setUp(self, n):
        self.n = n
        self.batch_size = 5
        self.diag = np.random.uniform(0.1, 1, (self.batch_size, self.n)).astype(
            np.float32
        )
        non_diag_size = self.n * (self.n - 1) // 2
        self.non_diag = np.random.uniform(
            -1, 1, (self.batch_size, non_diag_size)
        ).astype(np.float32)
        self.gy = np.random.uniform(-1, 1, (self.batch_size, self.n, self.n)).astype(
            np.float32
        )

    def check_forward(self, diag_data, non_diag_data, gpu):
        diag = torch.tensor(diag_data, requires_grad=True)
        non_diag = torch.tensor(non_diag_data, requires_grad=True)
        if gpu >= 0:
            diag = diag.to(torch.device("cuda", gpu))
            non_diag = non_diag.to(torch.device("cuda", gpu))

        y = lower_triangular_matrix(diag, non_diag)

        correct_y = np.zeros((self.batch_size, self.n, self.n), dtype=np.float32)

        tril_rows, tril_cols = np.tril_indices(self.n, -1)
        correct_y[:, tril_rows, tril_cols] = non_diag_data

        diag_rows, diag_cols = np.diag_indices(self.n)
        correct_y[:, diag_rows, diag_cols] = diag_data

        y = y.cpu() if gpu >= 0 else y
        np.testing.assert_allclose(correct_y, y.detach().numpy())

    def test_forward_cpu(self):
        self.check_forward(self.diag, self.non_diag, gpu=-1)

    @pytest.mark.gpu
    def test_forward_gpu(self):
        self.check_forward(self.diag, self.non_diag, gpu=0)

    def check_backward(self, diag_data, non_diag_data, gpu):
        diag = torch.tensor(diag_data, requires_grad=True)
        non_diag = torch.tensor(non_diag_data, requires_grad=True)
        if gpu >= 0:
            diag = diag.to(torch.device("cuda", gpu))
            non_diag = non_diag.to(torch.device("cuda", gpu))
        torch.autograd.gradcheck(
            lower_triangular_matrix, (diag, non_diag), eps=1e-02, rtol=1e-2
        )

    def test_backward_cpu(self):
        self.check_backward(self.diag, self.non_diag, gpu=-1)

    @pytest.mark.gpu
    def test_backward_gpu(self):
        self.check_backward(self.diag, self.non_diag, gpu=0)
