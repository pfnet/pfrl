import tempfile
import unittest
from unittest import mock

import pytest

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

    def test_unsupported_evaluation_hook(self):
        class UnsupportedEvaluationHook(
            pfrl.experiments.evaluation_hooks.EvaluationHook
        ):
            support_train_agent = False
            support_train_agent_batch = True
            support_train_agent_async = True

            def __call__(
                self,
                env,
                agent,
                evaluator,
                step,
                eval_stats,
                agent_stats,
                env_stats,
            ):
                pass

        unsupported_evaluation_hook = UnsupportedEvaluationHook()

        with pytest.raises(ValueError) as exception:
            pfrl.experiments.train_agent_with_evaluation(
                agent=mock.Mock(),
                env=mock.Mock(),
                steps=1,
                eval_n_steps=1,
                eval_n_episodes=None,
                eval_interval=1,
                outdir=mock.Mock(),
                evaluation_hooks=[unsupported_evaluation_hook],
            )

        assert str(
            exception.value
        ) == "{} does not support train_agent_with_evaluation().".format(
            unsupported_evaluation_hook
        )


@pytest.mark.parametrize("eval_during_episode", [False, True])
def test_eval_during_episode(eval_during_episode):

    outdir = tempfile.mkdtemp()

    agent = mock.MagicMock()
    env = mock.Mock()
    # Two episodes
    env.reset.side_effect = [("state", 0)] * 2
    env.step.side_effect = [
        (("state", 1), 0, False, {}),
        (("state", 2), 0, False, {}),
        (("state", 3), -0.5, True, {}),
        (("state", 4), 0, False, {}),
        (("state", 5), 1, True, {}),
    ]

    evaluator = mock.Mock()
    pfrl.experiments.train_agent(
        agent=agent,
        env=env,
        steps=5,
        outdir=outdir,
        evaluator=evaluator,
        eval_during_episode=eval_during_episode,
    )

    if eval_during_episode:
        # Must be called every timestep
        assert evaluator.evaluate_if_necessary.call_count == 5
        for i, call in enumerate(evaluator.evaluate_if_necessary.call_args_list):
            kwargs = call[1]
            assert i + 1 == kwargs["t"]
            assert kwargs["episodes"] == int(i >= 2) + int(i >= 4)
    else:
        # Must be called after every episode
        assert evaluator.evaluate_if_necessary.call_count == 2
        first_kwargs = evaluator.evaluate_if_necessary.call_args_list[0][1]
        second_kwargs = evaluator.evaluate_if_necessary.call_args_list[1][1]
        assert first_kwargs["t"] == 3
        assert first_kwargs["episodes"] == 1
        assert second_kwargs["t"] == 5
        assert second_kwargs["episodes"] == 2
