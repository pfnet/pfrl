import tempfile
import unittest
from unittest import mock

import pfrl


class TestTrainAgent(unittest.TestCase):
    def test(self):

        outdir = tempfile.mkdtemp()

        agent = mock.Mock()
        env = mock.Mock()
        # Reaches the terminal state after five actions
        env.reset.side_effect = [("state", 0)]
        env.step.side_effect = [
            (("state", 1), 0, False, {}),
            (("state", 2), 0, False, {}),
            (("state", 3), -0.5, False, {}),
            (("state", 4), 0, False, {}),
            (("state", 5), 1, True, {}),
        ]
        hook = mock.Mock()

        eval_stats_history = pfrl.experiments.train_agent(
            agent=agent, env=env, steps=5, outdir=outdir, step_hooks=[hook]
        )

        # No evaluation invoked when evaluator=None (default) is passed to train_agent.
        self.assertListEqual(eval_stats_history, [])

        self.assertEqual(agent.act.call_count, 5)
        self.assertEqual(agent.observe.call_count, 5)
        # done=True at state 5
        self.assertTrue(agent.observe.call_args_list[4][0][2])

        self.assertEqual(env.reset.call_count, 1)
        self.assertEqual(env.step.call_count, 5)

        self.assertEqual(hook.call_count, 5)
        # A hook receives (env, agent, step)
        for i, call in enumerate(hook.call_args_list):
            args, kwargs = call
            self.assertEqual(args[0], env)
            self.assertEqual(args[1], agent)
            # step starts with 1
            self.assertEqual(args[2], i + 1)

    def test_needs_reset(self):

        outdir = tempfile.mkdtemp()

        agent = mock.Mock()
        env = mock.Mock()
        # First episode: 0 -> 1 -> 2 -> 3 (reset)
        # Second episode: 4 -> 5 -> 6 -> 7 (done)
        env.reset.side_effect = [("state", 0), ("state", 4)]
        env.step.side_effect = [
            (("state", 1), 0, False, {}),
            (("state", 2), 0, False, {}),
            (("state", 3), 0, False, {"needs_reset": True}),
            (("state", 5), -0.5, False, {}),
            (("state", 6), 0, False, {}),
            (("state", 7), 1, True, {}),
        ]
        hook = mock.Mock()

        eval_stats_history = pfrl.experiments.train_agent(
            agent=agent, env=env, steps=5, outdir=outdir, step_hooks=[hook]
        )

        # No evaluation invoked when evaluator=None (default) is passed to train_agent.
        self.assertListEqual(eval_stats_history, [])

        self.assertEqual(agent.act.call_count, 5)
        self.assertEqual(agent.observe.call_count, 5)
        # done=False and reset=True at state 3
        self.assertFalse(agent.observe.call_args_list[2][0][2])
        self.assertTrue(agent.observe.call_args_list[2][0][3])

        self.assertEqual(env.reset.call_count, 2)
        self.assertEqual(env.step.call_count, 5)

        self.assertEqual(hook.call_count, 5)
        # A hook receives (env, agent, step)
        for i, call in enumerate(hook.call_args_list):
            args, kwargs = call
            self.assertEqual(args[0], env)
            self.assertEqual(args[1], agent)
            # step starts with 1
            self.assertEqual(args[2], i + 1)

    def test_with_evaluation_hooks(self):

        outdir = tempfile.mkdtemp()

        agent = mock.Mock()
        env = mock.Mock()
        # Reaches the terminal state after five actions
        env.reset.side_effect = [("state", 0)]
        env.step.side_effect = [
            (("state", 1), 0, False, {}),
            (("state", 2), 0, False, {}),
            (("state", 3), -0.5, False, {}),
            (("state", 4), 0, False, {}),
            (("state", 5), 1, True, {}),
        ]
        hook = mock.Mock()

        n_resets = 1
        dummy_stats = [
            ("average_q", 3.14),
            ("average_loss", 2.7),
            ("cumulative_steps", 42),
            ("n_updates", 8),
            ("rlen", 1),
        ]
        agent.get_statistics.side_effect = [dummy_stats] * n_resets

        evaluator = mock.Mock()
        # evaluator.evaluate_if_necessary is invoked after episode ends
        # (when evaluator is not None for train_agent)
        dummy_eval_score = 42
        evaluator.evaluate_if_necessary.return_value = dummy_eval_score

        evaluation_hook = mock.Mock()

        eval_stats_history = pfrl.experiments.train_agent(
            agent=agent,
            env=env,
            steps=5,
            outdir=outdir,
            step_hooks=[hook],
            evaluator=evaluator,
            evaluation_hooks=[evaluation_hook],
        )

        expected = [
            dict(**dict(dummy_stats), eval_score=dummy_eval_score)
            for _ in range(n_resets)
        ]
        self.assertListEqual(eval_stats_history, expected)

        self.assertEqual(agent.act.call_count, 5)
        self.assertEqual(agent.observe.call_count, 5)
        # done=True at state 5
        self.assertTrue(agent.observe.call_args_list[4][0][2])

        self.assertEqual(env.reset.call_count, 1)
        self.assertEqual(env.step.call_count, 5)

        self.assertEqual(hook.call_count, 5)
        # each hook receives (env, agent, step)
        for i, call in enumerate(hook.call_args_list):
            args, kwargs = call
            self.assertEqual(args[0], env)
            self.assertEqual(args[1], agent)
            # step starts with 1
            self.assertEqual(args[2], i + 1)

        # evaluation_hook receives (env, agent, evaluator, t, eval_score)
        self.assertEqual(evaluation_hook.call_count, n_resets)
        args = evaluation_hook.call_args[0]
        self.assertIs(args[0], env)
        self.assertIs(args[1], agent)
        self.assertIs(args[2], evaluator)
        self.assertIs(args[3], 5)
        self.assertIs(args[4], 42)
