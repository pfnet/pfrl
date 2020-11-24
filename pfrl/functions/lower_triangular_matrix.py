import numpy as np
import torch


def set_batch_non_diagonal(array, non_diag_val):
    batch_size, m, n = array.shape
    assert m == n
    rows, cols = np.tril_indices(n, -1)
    array[:, rows, cols] = non_diag_val


def set_batch_diagonal(array, diag_val):
    batch_size, m, n = array.shape
    assert m == n
    rows, cols = np.diag_indices(n)
    array[:, rows, cols] = diag_val


def lower_triangular_matrix(diag, non_diag):
    assert isinstance(diag, torch.Tensor)
    assert isinstance(non_diag, torch.Tensor)
    batch_size = diag.shape[0]
    n = diag.shape[1]
    y = torch.zeros((batch_size, n, n), dtype=torch.float32)
    y = y.to(diag.device)
    set_batch_non_diagonal(y, non_diag)
    set_batch_diagonal(y, diag)
    return y
