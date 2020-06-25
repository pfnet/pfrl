import logging
import unittest

from pfrl.explorers import epsilon_greedy


class TestEpsilonGreedy(unittest.TestCase):
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

        self.assertAlmostEqual(explorer.epsilon, 1.0)

        for t in range(100):
            explorer.select_action(t, greedy_action_func)

        self.assertAlmostEqual(explorer.epsilon, 0.1)

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

        self.assertAlmostEqual(explorer.epsilon, 0.1)

        for t in range(100):
            explorer.select_action(t, greedy_action_func)

        self.assertAlmostEqual(explorer.epsilon, 0.1)
