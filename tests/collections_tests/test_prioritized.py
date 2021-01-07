import random

import numpy as np
import pytest

from pfrl.collections import prioritized


@pytest.mark.parametrize("uniform_ratio", [0, 0.7, 1])
def test_prioritized_buffer_convergence(uniform_ratio):
    expected_corr_range = {0: (0.9, 1), 0.7: (0.5, 0.85), 1: (-0.3, 0.3)}[uniform_ratio]
    size = 100

    buf = prioritized.PrioritizedBuffer(capacity=size)
    for x in range(size):
        buf.append(x)

    priority_init = list([(i + 1) / size for i in range(size)])
    random.shuffle(priority_init)
    count_sampled = [0] * size

    def priority(x, n):
        if n == 0:
            return 1.0
        else:
            return priority_init[x] / count_sampled[x]

    for _ in range(200):
        sampled, probabilities, _ = buf.sample(16, uniform_ratio=uniform_ratio)
        priority_old = [priority(x, count_sampled[x]) for x in sampled]
        if uniform_ratio == 0:
            # assert: probabilities \propto priority_old
            qs = [x / y for x, y in zip(probabilities, priority_old)]
            for q in qs:
                np.testing.assert_allclose(q, qs[0])
        elif uniform_ratio == 1:
            # assert: uniform
            for p in probabilities:
                np.testing.assert_allclose(p, probabilities[0])
        for x in sampled:
            count_sampled[x] += 1
        priority_new = [priority(x, count_sampled[x]) for x in sampled]
        buf.set_last_priority(priority_new)

    for cnt in count_sampled:
        assert cnt >= 1

    corr = np.corrcoef(np.array([priority_init, count_sampled]))[0, 1]
    corr_lb, corr_ub = expected_corr_range
    assert corr > corr_lb
    assert corr < corr_ub


@pytest.mark.parametrize("capacity", [1, 10])
@pytest.mark.parametrize("wait_priority_after_sampling", [True, False])
@pytest.mark.parametrize("initial_priority", [0.1, 1])
@pytest.mark.parametrize("uniform_ratio", [0, 0.1, 1])
def test_prioritized_buffer_flood(
    capacity, wait_priority_after_sampling, initial_priority, uniform_ratio
):
    buf = prioritized.PrioritizedBuffer(
        capacity=capacity,
        wait_priority_after_sampling=wait_priority_after_sampling,
    )
    for _ in range(100):
        for x in range(capacity + 1):
            if wait_priority_after_sampling:
                buf.append(x)
            else:
                buf.append(x, priority=initial_priority)
        for _ in range(5):
            n = random.randrange(1, capacity + 1)
            buf.sample(n, uniform_ratio=uniform_ratio)
            if wait_priority_after_sampling:
                buf.set_last_priority([1.0] * n)
