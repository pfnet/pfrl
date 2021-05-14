import tempfile
import unittest
from unittest import mock

import numpy as np
import pytest

import pfrl
from pfrl.experiments import evaluator


@pytest.mark.parametrize("save_best_so_far_agent", [True, False])
@pytest.mark.parametrize("n_steps", [None, 1, 2])
@pytest.mark.parametrize("n_episodes", [None, 1, 2])
def test_evaluator_evaluate_if_necessary(save_best_so_far_agent, n_steps, n_episodes):

    outdir = tempfile.mkdtemp()

    # MagicMock can mock eval_mode while Mock cannot
    agent = mock.MagicMock()
    agent.act.return_value = "action"
    agent.get_statistics.return_value = []

    env = mock.Mock()
    env.reset.return_value = "obs"
    env.step.return_value = ("obs", 0, True, {})
    env.get_statistics.return_value = []

    evaluation_hook = mock.create_autospec(
        spec=pfrl.experiments.evaluation_hooks.EvaluationHook
    )

    either_none = (n_steps is None) != (n_episodes is None)
    if not either_none:
        with pytest.raises(AssertionError):
            agent_evaluator = evaluator.Evaluator(
                agent=agent,
                env=env,
                n_steps=n_steps,
                n_episodes=n_episodes,
                eval_interval=3,
                outdir=outdir,
                max_episode_len=None,
                step_offset=0,
                evaluation_hooks=[evaluation_hook],
                save_best_so_far_agent=save_best_so_far_agent,
            )
    else:
        value = n_steps or n_episodes
        agent_evaluator = evaluator.Evaluator(
            agent=agent,
            env=env,
            n_steps=n_steps,
            n_episodes=n_episodes,
            eval_interval=3,
            outdir=outdir,
            max_episode_len=None,
            step_offset=0,
            evaluation_hooks=[evaluation_hook],
            save_best_so_far_agent=save_best_so_far_agent,
        )

        agent_evaluator.evaluate_if_necessary(t=1, episodes=1)
        assert agent.act.call_count == 0
        assert evaluation_hook.call_count == 0

        agent_evaluator.evaluate_if_necessary(t=2, episodes=2)
        assert agent.act.call_count == 0
        assert evaluation_hook.call_count == 0

        # First evaluation
        agent_evaluator.evaluate_if_necessary(t=3, episodes=3)
        assert agent.act.call_count == value
        assert agent.observe.call_count == value
        assert evaluation_hook.call_count == 1
        if save_best_so_far_agent:
            assert agent.save.call_count == 1
        else:
            assert agent.save.call_count == 0

        # Second evaluation with the same score
        agent_evaluator.evaluate_if_necessary(t=6, episodes=6)
        assert agent.act.call_count == 2 * value
        assert agent.observe.call_count == 2 * value
        assert evaluation_hook.call_count == 2
        if save_best_so_far_agent:
            assert agent.save.call_count == 1
        else:
            assert agent.save.call_count == 0

        # Third evaluation with a better score
        env.step.return_value = ("obs", 1, True, {})
        agent_evaluator.evaluate_if_necessary(t=9, episodes=9)
        assert agent.act.call_count == 3 * value
        assert agent.observe.call_count == 3 * value
        assert evaluation_hook.call_count == 3
        if save_best_so_far_agent:
            assert agent.save.call_count == 2
        else:
            assert agent.save.call_count == 0


@pytest.mark.parametrize("save_best_so_far_agent", [True, False])
@pytest.mark.parametrize("n_episodes", [1, 2])
def test_async_evaluator_evaluate_if_necessary(save_best_so_far_agent, n_episodes):
    outdir = tempfile.mkdtemp()

    # MagicMock can mock eval_mode while Mock cannot
    agent = mock.MagicMock()
    agent.act.return_value = "action"
    agent.get_statistics.return_value = []

    env = mock.Mock()
    env.reset.return_value = "obs"
    env.step.return_value = ("obs", 0, True, {})
    env.get_statistics.return_value = []

    evaluation_hook = mock.create_autospec(
        spec=pfrl.experiments.evaluation_hooks.EvaluationHook
    )

    agent_evaluator = evaluator.AsyncEvaluator(
        n_steps=None,
        n_episodes=n_episodes,
        eval_interval=3,
        outdir=outdir,
        max_episode_len=None,
        step_offset=0,
        evaluation_hooks=[evaluation_hook],
        save_best_so_far_agent=save_best_so_far_agent,
    )

    agent_evaluator.evaluate_if_necessary(t=1, episodes=1, env=env, agent=agent)
    assert agent.act.call_count == 0
    assert evaluation_hook.call_count == 0

    agent_evaluator.evaluate_if_necessary(t=2, episodes=2, env=env, agent=agent)
    assert agent.act.call_count == 0
    assert evaluation_hook.call_count == 0

    # First evaluation
    agent_evaluator.evaluate_if_necessary(t=3, episodes=3, env=env, agent=agent)
    assert agent.act.call_count == n_episodes
    assert agent.observe.call_count == n_episodes
    assert evaluation_hook.call_count == 1
    if save_best_so_far_agent:
        assert agent.save.call_count == 1
    else:
        assert agent.save.call_count == 0

    # Second evaluation with the same score
    agent_evaluator.evaluate_if_necessary(t=6, episodes=6, env=env, agent=agent)
    assert agent.act.call_count == 2 * n_episodes
    assert agent.observe.call_count == 2 * n_episodes
    assert evaluation_hook.call_count == 2
    if save_best_so_far_agent:
        assert agent.save.call_count == 1
    else:
        assert agent.save.call_count == 0

    # Third evaluation with a better score
    env.step.return_value = ("obs", 1, True, {})
    agent_evaluator.evaluate_if_necessary(t=9, episodes=9, env=env, agent=agent)
    assert agent.act.call_count == 3 * n_episodes
    assert agent.observe.call_count == 3 * n_episodes
    assert evaluation_hook.call_count == 3
    if save_best_so_far_agent:
        assert agent.save.call_count == 2
    else:
        assert agent.save.call_count == 0


@pytest.mark.parametrize("n_episodes", [None, 1])
@pytest.mark.parametrize("n_steps", [2, 5, 6])
def test_run_evaluation_episodes_with_n_steps(n_episodes, n_steps):
    # MagicMock can mock eval_mode while Mock cannot
    agent = mock.MagicMock()
    env = mock.Mock()
    # First episode: 0 -> 1 -> 2 -> 3 (reset)
    # Second episode: 4 -> 5 -> 6 -> 7 (done)
    env.reset.side_effect = [("state", 0), ("state", 4)]
    env.step.side_effect = [
        (("state", 1), 0.1, False, {}),
        (("state", 2), 0.2, False, {}),
        (("state", 3), 0.3, False, {"needs_reset": True}),
        (("state", 5), -0.5, False, {}),
        (("state", 6), 0, False, {}),
        (("state", 7), 1, True, {}),
    ]

    if n_episodes:
        with pytest.raises(AssertionError):
            scores, lengths = evaluator.run_evaluation_episodes(
                env, agent, n_steps=n_steps, n_episodes=n_episodes
            )
    else:
        scores, lengths = evaluator.run_evaluation_episodes(
            env, agent, n_steps=n_steps, n_episodes=n_episodes
        )
        assert agent.act.call_count == n_steps
        assert agent.observe.call_count == n_steps
        if n_steps == 2:
            assert len(scores) == 1
            assert len(lengths) == 1
            np.testing.assert_allclose(scores[0], 0.3)
            np.testing.assert_allclose(lengths[0], 2)
        elif n_steps == 5:
            assert len(scores) == 1
            assert len(lengths) == 1
            np.testing.assert_allclose(scores[0], 0.6)
            np.testing.assert_allclose(lengths[0], 3)
        else:
            assert len(scores) == 2
            assert len(lengths) == 2
            np.testing.assert_allclose(scores[0], 0.6)
            np.testing.assert_allclose(scores[1], 0.5)
            np.testing.assert_allclose(lengths[0], 3)
            np.testing.assert_allclose(lengths[1], 3)


class TestRunEvaluationEpisode(unittest.TestCase):
    def test_needs_reset(self):
        # MagicMock can mock eval_mode while Mock cannot
        agent = mock.MagicMock()
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
        scores, lengths = evaluator.run_evaluation_episodes(
            env, agent, n_steps=None, n_episodes=2
        )
        assert len(scores) == 2
        assert len(lengths) == 2

        np.testing.assert_allclose(scores[0], 0)
        np.testing.assert_allclose(scores[1], 0.5)
        np.testing.assert_allclose(lengths[0], 3)
        np.testing.assert_allclose(lengths[1], 3)
        assert agent.act.call_count == 6
        assert agent.observe.call_count == 6


@pytest.mark.parametrize("n_episodes", [None, 1])
@pytest.mark.parametrize("n_steps", [2, 5, 6])
def test_batch_run_evaluation_episodes_with_n_steps(n_episodes, n_steps):
    # MagicMock can mock eval_mode while Mock cannot
    agent = mock.MagicMock()
    agent.batch_act.side_effect = [[1, 1]] * 5

    def make_env(idx):
        env = mock.Mock()
        if idx == 0:
            # First episode: 0 -> 1 -> 2 -> 3 (reset)
            # Second episode: 4 -> 5 -> 6 -> 7 (done)
            env.reset.side_effect = [("state", 0), ("state", 4)]
            env.step.side_effect = [
                (("state", 1), 0, False, {}),
                (("state", 2), 0.1, False, {}),
                (("state", 3), 0.2, False, {"needs_reset": True}),
                (("state", 5), -0.5, False, {}),
                (("state", 6), 0, False, {}),
                (("state", 7), 1, True, {}),
            ]
        else:
            # First episode: 0 -> 1 (reset)
            # Second episode: 2 -> 3 (reset)
            # Third episode: 4 -> 5 -> 6 -> 7 (done)
            env.reset.side_effect = [("state", 0), ("state", 2), ("state", 4)]
            env.step.side_effect = [
                (("state", 1), 2, False, {"needs_reset": True}),
                (("state", 3), 3, False, {"needs_reset": True}),
                (("state", 5), -0.6, False, {}),
                (("state", 6), 0, False, {}),
                (("state", 7), 1, True, {}),
            ]
        return env

    vec_env = pfrl.envs.SerialVectorEnv([make_env(i) for i in range(2)])
    if n_episodes:
        with pytest.raises(AssertionError):
            scores, lengths = evaluator.batch_run_evaluation_episodes(
                vec_env, agent, n_steps=n_steps, n_episodes=n_episodes
            )
    else:
        # First Env:  [1   2   (3_a)  5  6   (7_a)]
        # Second Env: [(1)(3_b) 5     6 (7_b)]
        scores, lengths = evaluator.batch_run_evaluation_episodes(
            vec_env, agent, n_steps=n_steps, n_episodes=n_episodes
        )
        if n_steps == 2:
            assert len(scores) == 1
            assert len(lengths) == 1
            np.testing.assert_allclose(scores[0], 0.1)
            np.testing.assert_allclose(lengths[0], 2)
            assert agent.batch_observe.call_count == 2
        else:
            assert len(scores) == 3
            assert len(lengths) == 3
            np.testing.assert_allclose(scores[0], 0.3)
            np.testing.assert_allclose(scores[1], 2.0)
            np.testing.assert_allclose(scores[2], 3.0)
            np.testing.assert_allclose(lengths[0], 3)
            np.testing.assert_allclose(lengths[1], 1)
            np.testing.assert_allclose(lengths[2], 1)
        # batch_reset should be all True
        assert all(agent.batch_observe.call_args[0][3])


class TestBatchRunEvaluationEpisode(unittest.TestCase):
    def test_needs_reset(self):
        # MagicMock can mock eval_mode while Mock cannot
        agent = mock.MagicMock()
        agent.batch_act.side_effect = [[1, 1]] * 5

        def make_env(idx):
            env = mock.Mock()
            if idx == 0:
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
            else:
                # First episode: 0 -> 1 (reset)
                # Second episode: 2 -> 3 (reset)
                # Third episode: 4 -> 5 -> 6 -> 7 (done)
                env.reset.side_effect = [("state", 0), ("state", 2), ("state", 4)]
                env.step.side_effect = [
                    (("state", 1), 2, False, {"needs_reset": True}),
                    (("state", 3), 3, False, {"needs_reset": True}),
                    (("state", 5), -0.6, False, {}),
                    (("state", 6), 0, False, {}),
                    (("state", 7), 1, True, {}),
                ]
            return env

        vec_env = pfrl.envs.SerialVectorEnv([make_env(i) for i in range(2)])

        # First Env: [1 2 (3_a) 5 6 (7_a)]
        # Second Env: [(1) (3_b) 5 6 (7_b)]
        # Results: (1), (3a), (3b), (7b)
        scores, lengths = evaluator.batch_run_evaluation_episodes(
            vec_env, agent, n_steps=None, n_episodes=4
        )
        assert len(scores) == 4
        assert len(lengths) == 4
        np.testing.assert_allclose(scores[0], 0)
        np.testing.assert_allclose(scores[1], 2)
        np.testing.assert_allclose(scores[2], 3)
        np.testing.assert_allclose(scores[3], 0.4)
        np.testing.assert_allclose(lengths[0], 3)
        np.testing.assert_allclose(lengths[1], 1)
        np.testing.assert_allclose(lengths[2], 1)
        np.testing.assert_allclose(lengths[3], 3)
        # batch_reset should be all True
        assert all(agent.batch_observe.call_args[0][3])
