import logging

import pytest

from pfrl.explorers import epsilon_greedy


@pytest.mark.parametrize("start_epsilon", [1.0, 0.5])
@pytest.mark.parametrize("end_epsilon", [0.5, 0.1])
@pytest.mark.parametrize("decay", [0.99, 0.1])
@pytest.mark.parametrize("steps", [1, 100])
class TestExponentialDecayEpsilonGreedy:
    @pytest.fixture(autouse=True)
    def setUp(self, steps, decay, end_epsilon, start_epsilon):
        self.steps = steps
        self.decay = decay
        self.end_epsilon = end_epsilon
        self.start_epsilon = start_epsilon

    def test(self):
        random_action_func_count = [0]
        greedy_action_func_count = [0]

        def random_action_func():
            random_action_func_count[0] += 1
            return 0

        def greedy_action_func():
            greedy_action_func_count[0] += 1
            return 0

        explorer = epsilon_greedy.ExponentialDecayEpsilonGreedy(
            self.start_epsilon, self.end_epsilon, self.decay, random_action_func
        )

        explorer.logger.addHandler(logging.StreamHandler())
        explorer.logger.setLevel(logging.DEBUG)

        assert pytest.approx(explorer.epsilon) == self.start_epsilon

        for t in range(self.steps):
            explorer.select_action(t, greedy_action_func)

        assert random_action_func_count[0] + greedy_action_func_count[0] == self.steps

        expected = max(
            self.start_epsilon * (self.decay ** (self.steps - 1)), self.end_epsilon
        )
        assert pytest.approx(explorer.epsilon) == expected


class TestEpsilonGreedy:
    def test_linear_decay_epsilon_greedy(self):

        random_action_func_count = [0]
        greedy_action_func_count = [0]

        def random_action_func():
            random_action_func_count[0] += 1
            return 0

        def greedy_action_func():
            greedy_action_func_count[0] += 1
            return 0

        explorer = epsilon_greedy.LinearDecayEpsilonGreedy(
            1.0, 0.1, 50, random_action_func
        )

        explorer.logger.addHandler(logging.StreamHandler())
        explorer.logger.setLevel(logging.DEBUG)

        assert pytest.approx(explorer.epsilon) == 1.0

        for t in range(100):
            explorer.select_action(t, greedy_action_func)

        assert random_action_func_count[0] + greedy_action_func_count[0] == 100

        assert pytest.approx(explorer.epsilon) == 0.1

    def test_constant_epsilon_greedy(self):

        random_action_func_count = [0]
        greedy_action_func_count = [0]

        def random_action_func():
            random_action_func_count[0] += 1
            return 0

        def greedy_action_func():
            greedy_action_func_count[0] += 1
            return 0

        explorer = epsilon_greedy.ConstantEpsilonGreedy(0.1, random_action_func)

        explorer.logger.addHandler(logging.StreamHandler())
        explorer.logger.setLevel(logging.DEBUG)

        assert pytest.approx(explorer.epsilon) == 0.1

        for t in range(100):
            explorer.select_action(t, greedy_action_func)

        assert random_action_func_count[0] + greedy_action_func_count[0] == 100

        assert pytest.approx(explorer.epsilon) == 0.1
