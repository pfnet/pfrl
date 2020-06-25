import torch


def conjugate_gradient(A_product_func, b, tol=1e-10, max_iter=10):
    """Conjugate Gradient (CG) method.

    This function solves Ax=b for the vector x, where A is a real
    positive-definite matrix and b is a real vector.

    Args:
        A_product_func (callable): Callable that returns the product of the
            matrix A and a given vector.
        b (numpy.ndarray or cupy.ndarray): The vector b.
        tol (float): Tolerance parameter for early stopping.
        max_iter (int): Maximum number of iterations.

    Returns:
        numpy.ndarray or cupy.ndarray: The solution.
            The array module will be the same as the argument b's.
    """
    x = torch.zeros_like(b)
    r0 = b - A_product_func(x)
    p = r0
    for _ in range(max_iter):
        a = torch.matmul(r0, r0) / torch.matmul(A_product_func(p), p)
        x = x + p * a
        r1 = r0 - A_product_func(p) * a
        if torch.norm(r1) < tol:
            return x
        b = torch.matmul(r1, r1) / torch.matmul(r0, r0)
        p = r1 + b * p
        r0 = r1
    return x
