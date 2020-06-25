import numpy as np
import pytest

from pfrl.explorers.additive_ou import AdditiveOU


@pytest.mark.parametrize("action_size", [1, 3])
@pytest.mark.parametrize("sigma_type", ["scalar", "ndarray"])
def test_additive_ou(action_size, sigma_type):
    def greedy_action_func():
        return np.asarray([0] * action_size, dtype=np.float32)

    if sigma_type == "scalar":
        sigma = np.random.rand()
    elif sigma_type == "ndarray":
        sigma = np.random.rand(action_size)
    theta = np.random.rand()

    explorer = AdditiveOU(theta=theta, sigma=sigma)

    print("theta:", theta, "sigma", sigma)
    for t in range(100):
        a = explorer.select_action(t, greedy_action_func)
        print(t, a)
