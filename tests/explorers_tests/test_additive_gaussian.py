import numpy as np
import pytest

from pfrl.explorers.additive_gaussian import AdditiveGaussian


@pytest.mark.parametrize("action_size", [1, 3])
@pytest.mark.parametrize("scale", [0, 0.1])
@pytest.mark.parametrize("low", [None, -0.4])
@pytest.mark.parametrize("high", [None, 0.4])
def test_additive_gaussian(action_size, scale, low, high):
    def greedy_action_func():
        return np.full(action_size, 0.3)

    explorer = AdditiveGaussian(scale, low=low, high=high)

    actions = []
    for t in range(100):
        a = explorer.select_action(t, greedy_action_func)

        if low is not None:
            # Clipped at lower edge
            assert (a >= low).all()

        if high is not None:
            # Clipped at upper edge
            assert (a <= high).all()

        if scale == 0:
            # Without noise
            assert (a == 0.3).all()
        else:
            # With noise
            assert not (a == 0.3).all()
        actions.append(a)

    if low is None and high is None:
        np.testing.assert_allclose(np.mean(np.asarray(actions), axis=0), 0.3, atol=0.1)
