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
        eval_score = 3.14

        optuna_pruner_hook(env, agent, evaluator, step, eval_score)

        trial.report.assert_called_once_with(eval_score, step)

    def test_should_prune(self):
        trial = Mock()
        trial.should_prune.return_value = True
        optuna_pruner_hook = pfrl.experiments.OptunaPrunerHook(trial=trial)

        env = Mock()
        agent = Mock()
        evaluator = Mock()
        step = 42
        eval_score = 3.14

        with self.assertRaises(optuna.TrialPruned):
            optuna_pruner_hook(env, agent, evaluator, step, eval_score)

        trial.report.assert_called()
        trial.report.assert_called_once_with(eval_score, step)
