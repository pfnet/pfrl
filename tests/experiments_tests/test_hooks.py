import unittest

import numpy as np

import pfrl


class TestLinearInterpolationHook(unittest.TestCase):
    def test_call(self):

        buf = []

        def setter(env, agent, value):
            buf.append(value)

        hook = pfrl.experiments.LinearInterpolationHook(
            total_steps=10, start_value=0.1, stop_value=1.0, setter=setter
        )

        for step in range(1, 10 + 1):
            hook(env=None, agent=None, step=step)

        np.testing.assert_allclose(buf, np.arange(1, 10 + 1, dtype=np.float32) / 10)
