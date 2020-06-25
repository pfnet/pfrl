import numpy as np


def sample_n_k(n, k):
    """Sample k distinct elements uniformly from range(n)"""

    if not 0 <= k <= n:
        raise ValueError("Sample larger than population or is negative")
    if k == 0:
        return np.empty((0,), dtype=np.int64)
    elif 3 * k >= n:
        return np.random.choice(n, k, replace=False)
    else:
        result = np.random.choice(n, 2 * k)
        selected = set()
        selected_add = selected.add
        j = k
        for i in range(k):
            x = result[i]
            while x in selected:
                x = result[i] = result[j]
                j += 1
                if j == 2 * k:
                    # This is slow, but it rarely happens.
                    result[k:] = np.random.choice(n, k)
                    j = k
            selected_add(x)
        return result[:k]
