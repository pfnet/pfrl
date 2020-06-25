import timeit
import unittest

import numpy as np
import pytest
from scipy import stats

from pfrl.utils.random import sample_n_k


@pytest.mark.parametrize(
    "n,k", [(2, 2), (5, 1), (5, 4), (7, 2), (20, 10), (100, 5), (1, 0), (0, 0)]
)
class TestSampleNK:
    @pytest.fixture(autouse=True)
    def setUp(self, n, k):
        self.n = n
        self.k = k

    def test_fast(self):
        self.samples = [sample_n_k(self.n, self.k) for _ in range(200)]
        self.subtest_constraints()

    def subtest_constraints(self):
        for s in self.samples:
            assert len(s) == self.k

            all(0 <= x < self.n for x in s)

            # distinct
            t = np.unique(s)
            assert len(t) == self.k

    @pytest.mark.slow
    def test_slow(self):
        self.samples = [sample_n_k(self.n, self.k) for _ in range(10000)]
        self.subtest_total_counts()
        self.subtest_order_counts()

    def subtest_total_counts(self):
        if self.k in [0, self.n]:
            return

        cnt = np.zeros(self.n)
        for s in self.samples:
            for x in s:
                cnt[x] += 1

        m = len(self.samples)

        p = self.k / self.n
        mean = m * p
        std = np.sqrt(m * p * (1 - p))

        self.subtest_normal_distrib(cnt, mean, std)

    def subtest_order_counts(self):
        if self.k < 2:
            return

        ordered_pairs = [(i, j) for j in range(self.k) for i in range(j)]
        cnt = np.zeros(len(ordered_pairs))

        for s in self.samples:
            for t, (i, j) in enumerate(ordered_pairs):
                if s[i] < s[j]:
                    cnt[t] += 1

        m = len(self.samples)

        mean = m / 2
        std = np.sqrt(m / 4)

        self.subtest_normal_distrib(cnt, mean, std)

    def subtest_normal_distrib(self, xs, mean, std):
        _, pvalue = stats.kstest(xs, "norm", (mean, std))
        assert pvalue > 1e-5


class TestSampleNKSpeed(unittest.TestCase):
    def get_timeit(self, setup):
        return min(
            timeit.Timer(
                "for n in range(64, 10000): sample_n_k(n, 64)", setup=setup
            ).repeat(repeat=10, number=1)
        )

    @pytest.mark.slow
    def _test(self):
        t = self.get_timeit("from pfrl.utils.random import sample_n_k")

        # faster than random.sample
        t1 = self.get_timeit(
            """
import random
def sample_n_k(n, k):
    return random.sample(range(n), k)
"""
        )
        self.assertLess(t, t1)

        # faster than np.random.choice(..., replace=False)
        t2 = self.get_timeit(
            """
import numpy as np
def sample_n_k(n, k):
    return np.random.choice(n, k, replace=False)
"""
        )
        self.assertLess(t, t2)
