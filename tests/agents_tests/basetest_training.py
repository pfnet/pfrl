import logging
import os
import tempfile
from unittest import mock

import numpy as np
import pytest

import pfrl
from pfrl.experiments import (
    train_agent_async,
    train_agent_batch_with_evaluation,
    train_agent_with_evaluation,
)
from pfrl.experiments.evaluator import (
    batch_run_evaluation_episodes,
    run_evaluation_episodes,
)
from pfrl.utils import random_seed


class _TestTraining:
    @pytest.fixture(autouse=True)
    def set_tmp_paths(self):
        self.tmpdir = tempfile.mkdtemp()
        self.agent_dirname = os.path.join(self.tmpdir, "agent_final")
        self.rbuf_filename = os.path.join(self.tmpdir, "rbuf.pkl")

    def make_agent(self, env, gpu):
        raise NotImplementedError()

    def make_env_and_successful_return(self, test):
        raise NotImplementedError()

    def _test_training(self, gpu, steps=5000, load_model=False, require_success=True):

        random_seed.set_random_seed(1)
        logging.basicConfig(level=logging.DEBUG)

        env = self.make_env_and_successful_return(test=False)[0]
        test_env, successful_return = self.make_env_and_successful_return(test=True)
        agent = self.make_agent(env, gpu)

        if load_model:
            print("Load agent from", self.agent_dirname)
            agent.load(self.agent_dirname)
            agent.replay_buffer.load(self.rbuf_filename)

        # Train
        train_agent_with_evaluation(
            agent=agent,
            env=env,
            steps=steps,
            outdir=self.tmpdir,
            eval_interval=200,
            eval_n_steps=None,
            eval_n_episodes=5,
            successful_score=1,
            eval_env=test_env,
        )

        # Test
        n_test_runs = 5
        eval_returns, _ = run_evaluation_episodes(
            test_env,
            agent,
            n_steps=None,
            n_episodes=n_test_runs,
        )
        n_succeeded = np.sum(np.asarray(eval_returns) >= successful_return)
        if require_success:
            assert n_succeeded == n_test_runs

        # Save
        agent.save(self.agent_dirname)
        agent.replay_buffer.save(self.rbuf_filename)

    @pytest.mark.slow
    @pytest.mark.gpu
    def test_training_gpu(self):
        self._test_training(0, steps=100000)
        self._test_training(0, steps=0, load_model=True)

    @pytest.mark.slow
    def test_training_cpu(self):
        self._test_training(-1, steps=100000)
        self._test_training(-1, steps=0, load_model=True)

    @pytest.mark.gpu
    def test_training_gpu_fast(self):
        self._test_training(0, steps=10, require_success=False)
        self._test_training(0, steps=0, load_model=True, require_success=False)

    def test_training_cpu_fast(self):
        self._test_training(-1, steps=10, require_success=False)
        self._test_training(-1, steps=0, load_model=True, require_success=False)


class _TestBatchTrainingMixin(object):
    """Mixin for testing batch training.

    Inherit this after _TestTraining to enable test cases for batch training.
    """

    def make_vec_env_and_successful_return(self, test, num_envs=2):
        successful_return = self.make_env_and_successful_return(test=test)[1]
        vec_env = pfrl.envs.SerialVectorEnv(
            [self.make_env_and_successful_return(test=test)[0] for _ in range(num_envs)]
        )
        return vec_env, successful_return

    def _test_batch_training(
        self, gpu, steps=5000, load_model=False, require_success=True
    ):

        random_seed.set_random_seed(1)
        logging.basicConfig(level=logging.DEBUG)

        env, _ = self.make_vec_env_and_successful_return(test=False)
        test_env, successful_return = self.make_vec_env_and_successful_return(test=True)
        agent = self.make_agent(env, gpu)

        if load_model:
            print("Load agent from", self.agent_dirname)
            agent.load(self.agent_dirname)
            agent.replay_buffer.load(self.rbuf_filename)

        # Train
        train_agent_batch_with_evaluation(
            agent=agent,
            env=env,
            steps=steps,
            outdir=self.tmpdir,
            eval_interval=200,
            eval_n_steps=None,
            eval_n_episodes=5,
            successful_score=1,
            eval_env=test_env,
        )
        env.close()

        # Test
        n_test_runs = 5
        eval_returns, _ = batch_run_evaluation_episodes(
            test_env,
            agent,
            n_steps=None,
            n_episodes=n_test_runs,
        )
        test_env.close()
        n_succeeded = np.sum(np.asarray(eval_returns) >= successful_return)
        if require_success:
            assert n_succeeded == n_test_runs

        # Save
        agent.save(self.agent_dirname)
        agent.replay_buffer.save(self.rbuf_filename)

    @pytest.mark.slow
    @pytest.mark.gpu
    def test_batch_training_gpu(self):
        self._test_batch_training(0, steps=100000)
        self._test_batch_training(0, steps=0, load_model=True)

    @pytest.mark.slow
    def test_batch_training_cpu(self):
        self._test_batch_training(-1, steps=100000)
        self._test_batch_training(-1, steps=0, load_model=True)

    @pytest.mark.gpu
    def test_batch_training_gpu_fast(self):
        self._test_batch_training(0, steps=10, require_success=False)
        self._test_batch_training(0, steps=0, load_model=True, require_success=False)

    def test_batch_training_cpu_fast(self):
        self._test_batch_training(-1, steps=10, require_success=False)
        self._test_batch_training(-1, steps=0, load_model=True, require_success=False)


class _TestActorLearnerTrainingMixin(object):
    """Mixin for testing actor-learner training.
    Inherit this after _TestTraining to enable test cases for batch training.
    """

    def _test_actor_learner_training(self, gpu, steps=100000, require_success=True):

        logging.basicConfig(level=logging.DEBUG)

        test_env, successful_return = self.make_env_and_successful_return(test=True)
        agent = self.make_agent(test_env, gpu)

        # cumulative_steps init to 0
        assert agent.cumulative_steps == 0

        def make_env(process_idx, test):
            env, _ = self.make_env_and_successful_return(test=test)
            return env

        step_hook = mock.Mock()
        optimizer_step_hook = mock.Mock()

        # Train
        if steps > 0:
            (
                make_actor,
                learner,
                poller,
                exception_event,
            ) = agent.setup_actor_learner_training(
                n_actors=2,
                step_hooks=[step_hook],
                optimizer_step_hooks=[optimizer_step_hook],
            )

            poller.start()
            learner.start()
            train_agent_async(
                processes=2,
                steps=steps,
                outdir=self.tmpdir,
                eval_interval=200,
                eval_n_steps=None,
                eval_n_episodes=5,
                successful_score=successful_return,
                make_env=make_env,
                make_agent=make_actor,
                stop_event=learner.stop_event,
                exception_event=exception_event,
            )
            learner.stop()
            learner.join()
            poller.stop()
            poller.join()

        # Test

        # Because in actor-learner traininig the model can be updated between
        # evaluation and saving, it is difficult to guarantee the learned
        # model successfully passes the test.
        # Thus we only check if the training was successful.

        # As the test can finish before running all the steps,
        # we can only test the range
        assert agent.cumulative_steps > 0
        assert agent.cumulative_steps <= steps + 1

        # Unlike the non-actor-learner cases, the step_hooks and
        # optimizer_step_hooks are only called when the update happens
        # when we do a fast test, the update may not be triggered due to
        # limited amount of experience, the call_count can be 0 in such case
        assert step_hook.call_count >= 0
        assert step_hook.call_count <= steps / agent.update_interval
        assert optimizer_step_hook.call_count == step_hook.call_count

        for i, call in enumerate(step_hook.call_args_list):
            args, kwargs = call
            assert args[0] is None
            assert args[1] is agent
            assert args[2] == (i + 1) * agent.update_interval

        for i, call in enumerate(optimizer_step_hook.call_args_list):
            args, kwargs = call
            assert args[0] is None
            assert args[1] is agent
            assert args[2] == i + 1

        successful_path = os.path.join(self.tmpdir, "successful")
        finished_path = os.path.join(self.tmpdir, "{}_finish".format(steps))
        if require_success:
            assert os.path.exists(successful_path)
        else:
            assert os.path.exists(successful_path) or os.path.exists(finished_path)

    @pytest.mark.async_
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_actor_learner_training_gpu(self):
        self._test_actor_learner_training(0, steps=100000)

    @pytest.mark.async_
    @pytest.mark.slow
    def test_actor_learner_training_cpu(self):
        self._test_actor_learner_training(-1, steps=100000)

    @pytest.mark.async_
    @pytest.mark.gpu
    def test_actor_learner_training_gpu_fast(self):
        self._test_actor_learner_training(0, steps=10, require_success=False)

    @pytest.mark.async_
    def test_actor_learner_training_cpu_fast(self):
        self._test_actor_learner_training(-1, steps=10, require_success=False)
