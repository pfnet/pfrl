import os
import tempfile
import unittest
from unittest import mock

import pytest
import torch.multiprocessing as mp

import pfrl
from pfrl.experiments.train_agent_async import train_loop


@pytest.mark.parametrize("num_envs", [1, 2])
@pytest.mark.parametrize("max_episode_len", [None, 2])
def test_train_agent_async(num_envs, max_episode_len):

    steps = 50

    outdir = tempfile.mkdtemp()

    agent = mock.Mock()
    agent.shared_attributes = []

    def _make_env(process_idx, test):
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
            env.step.side_effect = [
                (("state", 1), 0, False, {}),
            ] * 1000
        return env

    # Keep references to mock envs to check their states later
    envs = [_make_env(i, test=False) for i in range(num_envs)]
    eval_envs = [_make_env(i, test=True) for i in range(num_envs)]

    def make_env(process_idx, test):
        if test:
            return eval_envs[process_idx]
        else:
            return envs[process_idx]

    # Mock states cannot be shared among processes. To check states of mock
    # objects, threading is used instead of multiprocessing.
    # Because threading.Thread does not have .exitcode attribute, we
    # add the attribute manually to avoid an exception.
    import threading

    # Mock.call_args_list does not seem thread-safe
    hook_lock = threading.Lock()
    hook = mock.Mock()

    def hook_locked(*args, **kwargs):
        with hook_lock:
            return hook(*args, **kwargs)

    with mock.patch(
        "torch.multiprocessing.Process", threading.Thread
    ), mock.patch.object(threading.Thread, "exitcode", create=True, new=0):
        pfrl.experiments.train_agent_async(
            processes=num_envs,
            agent=agent,
            make_env=make_env,
            steps=steps,
            outdir=outdir,
            max_episode_len=max_episode_len,
            global_step_hooks=[hook_locked],
        )

    if num_envs == 1:
        assert agent.act.call_count == steps
        assert agent.observe.call_count == steps
    elif num_envs > 1:
        assert agent.act.call_count > steps
        assert agent.observe.call_count == agent.act.call_count

    # All the envs (including eval envs) should to be closed
    for env in envs + eval_envs:
        assert env.close.call_count == 1

    if num_envs == 1:
        assert hook.call_count == steps
    elif num_envs > 1:
        assert hook.call_count > steps

    # A hook receives (env, agent, step)
    for i, call in enumerate(hook.call_args_list):
        args, kwargs = call
        assert any(args[0] == env for env in envs)
        assert args[1] == agent
        if num_envs == 1:
            # If num_envs == 1, a hook should be called sequentially.
            # step starts with 1
            assert args[2] == i + 1
    if num_envs > 1:
        # If num_envs > 1, a hook may not be called sequentially.
        # Thus, we only check if they are called for each step.
        hook_steps = [call[0][2] for call in hook.call_args_list]
        assert list(range(1, len(hook.call_args_list) + 1)) == sorted(hook_steps)

    # Agent should be saved
    agent.save.assert_called_once_with(os.path.join(outdir, "{}_finish".format(steps)))


def test_unsupported_evaluation_hook():
    class UnsupportedEvaluationHook(pfrl.experiments.evaluation_hooks.EvaluationHook):
        support_train_agent = True
        support_train_agent_batch = True
        support_train_agent_async = False

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
        pfrl.experiments.train_agent_async(
            outdir=mock.Mock(),
            processes=mock.Mock(),
            make_env=mock.Mock(),
            evaluation_hooks=[unsupported_evaluation_hook],
        )

    assert str(exception.value) == "{} does not support train_agent_async().".format(
        unsupported_evaluation_hook
    )


class TestTrainLoop(unittest.TestCase):
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

        counter = mp.Value("i", 0)
        episodes_counter = mp.Value("i", 0)
        stop_event = mp.Event()
        exception_event = mp.Event()
        train_loop(
            process_idx=0,
            env=env,
            agent=agent,
            steps=5,
            outdir=outdir,
            counter=counter,
            episodes_counter=episodes_counter,
            stop_event=stop_event,
            exception_event=exception_event,
        )

        self.assertEqual(agent.act.call_count, 5)
        self.assertEqual(agent.observe.call_count, 5)
        self.assertEqual(agent.observe.call_count, 5)
        # done=False and reset=True at state 3
        self.assertFalse(agent.observe.call_args_list[2][0][2])
        self.assertTrue(agent.observe.call_args_list[2][0][3])

        self.assertEqual(env.reset.call_count, 2)
        self.assertEqual(env.step.call_count, 5)
