import unittest
from unittest.mock import Mock

import optuna

import pfrl


class TestOptunaPrunerHook(unittest.TestCase):
    def test_dont_prune(self):
        trial = Mock()
        trial.should_prune.return_value = False
        optuna_pruner_hook = pfrl.experiments.OptunaPrunerHook(trial=trial)

        env = Mock()
        agent = Mock()
        evaluator = Mock()
        step = 42
        eval_stats = {"mean": 3.14}
        agent_stats = [("dummy", 2.7)]
        env_stats = []

        optuna_pruner_hook(
            env, agent, evaluator, step, eval_stats, agent_stats, env_stats
        )

        trial.report.assert_called_once_with(eval_stats["mean"], step)

    def test_should_prune(self):
        trial = Mock()
        trial.should_prune.return_value = True
        optuna_pruner_hook = pfrl.experiments.OptunaPrunerHook(trial=trial)

        env = Mock()
        agent = Mock()
        evaluator = Mock()
        step = 42
        eval_stats = {"mean": 3.14}
        agent_stats = [("dummy", 2.7)]
        env_stats = []

        with self.assertRaises(optuna.TrialPruned):
            optuna_pruner_hook(
                env, agent, evaluator, step, eval_stats, agent_stats, env_stats
            )

        trial.report.assert_called_once_with(eval_stats["mean"], step)
