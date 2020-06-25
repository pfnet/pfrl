import math
import tempfile
import unittest
from unittest import mock

import pytest

import pfrl


@pytest.mark.parametrize("num_envs", [1, 2])
@pytest.mark.parametrize("max_episode_len", [None, 2])
@pytest.mark.parametrize("steps", [5, 6])
def test_train_agent_batch(num_envs, max_episode_len, steps):

    outdir = tempfile.mkdtemp()

    agent = mock.Mock()
    agent.batch_act.side_effect = [[1] * num_envs] * 1000

    def make_env():
        env = mock.Mock()
        env.reset.side_effect = [("state", 0)] * 1000
        if max_episode_len is None:
            # Episodic env that terminates after 5 actions
            env.step.side_effect = [
                (("state", 1), 0, False, {}),
                (("state", 2), 0, False, {}),
                (("state", 3), -0.5, False, {}),
                (("state", 4), 0, False, {}),
                (("state", 5), 1, True, {}),
            ] * 1000
        else:
            # Continuing env
            env.step.side_effect = [(("state", 1), 0, False, {}),] * 1000
        return env

    vec_env = pfrl.envs.SerialVectorEnv([make_env() for _ in range(num_envs)])

    hook = mock.Mock()

    pfrl.experiments.train_agent_batch(
        agent=agent,
        env=vec_env,
        steps=steps,
        outdir=outdir,
        max_episode_len=max_episode_len,
        step_hooks=[hook],
    )

    iters = math.ceil(steps / num_envs)
    assert agent.batch_act.call_count == iters
    assert agent.batch_observe.call_count == iters

    for env in vec_env.envs:
        if max_episode_len is None:
            if num_envs == 1:
                if steps == 6:
                    # In the beginning and after 5 iterations
                    assert env.reset.call_count == 2
                else:
                    assert steps == 5
                    # Only in the beginning. While the last state is
                    # terminal, env.reset should not be called because
                    # training is complete.
                    assert env.reset.call_count == 1
            elif num_envs == 2:
                # Only in the beginning
                assert env.reset.call_count == 1
            else:
                assert False
        elif max_episode_len == 2:
            if num_envs == 1:
                # In the beginning, after 2 and 4 iterations
                assert env.reset.call_count == 3
            elif num_envs == 2:
                # In the beginning, after 2 iterations
                assert env.reset.call_count == 2
            else:
                assert False
        assert env.step.call_count == iters

    if steps % num_envs == 0:
        assert hook.call_count == steps
    else:
        assert hook.call_count == num_envs * iters

    # A hook receives (env, agent, step)
    for i, call in enumerate(hook.call_args_list):
        args, kwargs = call
        assert args[0] == vec_env
        assert args[1] == agent
        # step starts with 1
        assert args[2] == i + 1


class TestTrainAgentBatchNeedsReset(unittest.TestCase):
    def test_needs_reset(self):
        steps = 10

        outdir = tempfile.mkdtemp()

        agent = mock.Mock()
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
                    (("state", 1), 0, False, {"needs_reset": True}),
                    (("state", 3), 0, False, {"needs_reset": True}),
                    (("state", 5), -0.5, False, {}),
                    (("state", 6), 0, False, {}),
                    (("state", 7), 1, True, {}),
                ]
            return env

        vec_env = pfrl.envs.SerialVectorEnv([make_env(i) for i in range(2)])

        pfrl.experiments.train_agent_batch(
            agent=agent, env=vec_env, steps=steps, outdir=outdir,
        )

        self.assertEqual(vec_env.envs[0].reset.call_count, 2)
        self.assertEqual(vec_env.envs[0].step.call_count, 5)
        self.assertEqual(vec_env.envs[1].reset.call_count, 3)
        self.assertEqual(vec_env.envs[1].step.call_count, 5)
