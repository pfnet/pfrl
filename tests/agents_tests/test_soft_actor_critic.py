import os
import tempfile

import numpy as np
import pytest
import torch
from torch import distributions, nn

import pfrl
from pfrl.envs.abc import ABC
from pfrl.experiments import (
    train_agent_batch_with_evaluation,
    train_agent_with_evaluation,
)
from pfrl.experiments.evaluator import (
    batch_run_evaluation_episodes,
    run_evaluation_episodes,
)
from pfrl.nn.lmbda import Lambda


@pytest.mark.parametrize("episodic", [False, True])
class TestSoftActorCritic:
    @pytest.fixture(autouse=True)
    def setUp(self, episodic):
        self.episodic = episodic
        self.tmpdir = tempfile.mkdtemp()
        self.agent_dirname = os.path.join(self.tmpdir, "agent_final")

    @pytest.mark.slow
    def test_abc_cpu(self):
        print("thing")
        self._test_abc()
        self._test_abc(steps=0, load_model=True)

    @pytest.mark.slow
    @pytest.mark.gpu
    def test_abc_gpu(self):
        self._test_abc(gpu=0)

    def test_abc_fast_cpu(self):
        self._test_abc(steps=100, require_success=False)
        self._test_abc(steps=0, require_success=False, load_model=True)

    @pytest.mark.gpu
    def test_abc_fast_gpu(self):
        self._test_abc(steps=100, require_success=False, gpu=0)

    @pytest.mark.slow
    def test_abc_batch_cpu(self):
        self._test_abc_batch()
        self._test_abc_batch(steps=0, load_model=True)

    @pytest.mark.slow
    @pytest.mark.gpu
    def test_abc_batch_gpu(self):
        self._test_abc_batch(gpu=0)

    def test_abc_batch_fast_cpu(self):
        self._test_abc_batch(steps=100, require_success=False)
        self._test_abc_batch(steps=0, require_success=False, load_model=True)

    @pytest.mark.gpu
    def test_abc_batch_fast_gpu(self):
        self._test_abc_batch(steps=100, require_success=False, gpu=0)

    def _test_abc(self, steps=100000, require_success=True, gpu=-1, load_model=False):

        env, _ = self.make_env_and_successful_return(test=False)
        test_env, successful_return = self.make_env_and_successful_return(test=True)

        agent = self.make_agent(env, gpu)

        if load_model:
            print("Load agent from", self.agent_dirname)
            agent.load(self.agent_dirname)

        max_episode_len = None if self.episodic else 2

        # Train
        train_agent_with_evaluation(
            agent=agent,
            env=env,
            eval_env=test_env,
            steps=steps,
            outdir=self.tmpdir,
            eval_interval=200,
            eval_n_steps=None,
            eval_n_episodes=5,
            successful_score=successful_return,
            train_max_episode_len=max_episode_len,
        )

        # Test
        n_test_runs = 5
        eval_returns, _ = run_evaluation_episodes(
            test_env,
            agent,
            n_steps=None,
            n_episodes=n_test_runs,
            max_episode_len=max_episode_len,
        )
        if require_success:
            n_succeeded = np.sum(np.asarray(eval_returns) >= successful_return)
            assert n_succeeded == n_test_runs

        # Save
        agent.save(self.agent_dirname)

    def _test_abc_batch(
        self, steps=100000, require_success=True, gpu=-1, load_model=False
    ):

        env, _ = self.make_vec_env_and_successful_return(test=False)
        test_env, successful_return = self.make_vec_env_and_successful_return(test=True)

        agent = self.make_agent(env, gpu)

        if load_model:
            print("Load agent from", self.agent_dirname)
            agent.load(self.agent_dirname)

        max_episode_len = None if self.episodic else 2

        # Train
        train_agent_batch_with_evaluation(
            agent=agent,
            env=env,
            eval_env=test_env,
            steps=steps,
            outdir=self.tmpdir,
            eval_interval=200,
            eval_n_steps=None,
            eval_n_episodes=5,
            successful_score=successful_return,
            max_episode_len=max_episode_len,
        )
        env.close()

        # Test
        n_test_runs = 5
        eval_returns, _ = batch_run_evaluation_episodes(
            test_env,
            agent,
            n_steps=None,
            n_episodes=n_test_runs,
            max_episode_len=max_episode_len,
        )
        test_env.close()
        if require_success:
            n_succeeded = np.sum(np.asarray(eval_returns) >= successful_return)
            assert n_succeeded == n_test_runs

        # Save
        agent.save(self.agent_dirname)

    def make_agent(self, env, gpu):
        obs_size = env.observation_space.low.size
        action_size = env.action_space.low.size
        hidden_size = 20

        def squashed_diagonal_gaussian_head(x):
            assert x.shape[-1] == action_size * 2
            mean, log_scale = torch.split(x, int(list(x.size())[-1] / 2), dim=1)
            log_scale = torch.clamp(log_scale, -20.0, 2.0)
            var = torch.exp(log_scale * 2)
            base_distribution = distributions.Independent(
                distributions.Normal(loc=mean, scale=torch.sqrt(var)), 1
            )
            # cache_size=1 is required for numerical stability
            return distributions.transformed_distribution.TransformedDistribution(
                base_distribution,
                [distributions.transforms.TanhTransform(cache_size=1)],
            )

        policy = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(
                hidden_size,
                action_size * 2,
            ),
            nn.Tanh(),
            Lambda(squashed_diagonal_gaussian_head),
        )
        policy[2].weight.detach().mul_(1e-1)
        policy_optimizer = torch.optim.Adam(policy.parameters())

        def make_q_func_with_optimizer():
            q_func = nn.Sequential(
                pfrl.nn.ConcatObsAndAction(),
                nn.Linear(obs_size + action_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
            )
            q_func[3].weight.detach().mul_(1e-1)
            q_func_optimizer = torch.optim.Adam(q_func.parameters(), lr=1e-2)
            return q_func, q_func_optimizer

        q_func1, q_func1_optimizer = make_q_func_with_optimizer()
        q_func2, q_func2_optimizer = make_q_func_with_optimizer()

        rbuf = pfrl.replay_buffers.ReplayBuffer(10 ** 6)

        def burnin_action_func():
            return np.random.uniform(
                env.action_space.low, env.action_space.high
            ).astype(np.float32)

        agent = pfrl.agents.SoftActorCritic(
            policy=policy,
            q_func1=q_func1,
            q_func2=q_func2,
            policy_optimizer=policy_optimizer,
            q_func1_optimizer=q_func1_optimizer,
            q_func2_optimizer=q_func2_optimizer,
            replay_buffer=rbuf,
            gamma=0.5,
            minibatch_size=100,
            replay_start_size=100,
            burnin_action_func=burnin_action_func,
            entropy_target=-action_size,
            max_grad_norm=1.0,
        )

        return agent

    def make_env_and_successful_return(self, test):
        env = ABC(
            discrete=False,
            episodic=self.episodic or test,
            deterministic=test,
        )
        return env, 1

    def make_vec_env_and_successful_return(self, test, num_envs=3):
        def make_env():
            return self.make_env_and_successful_return(test)[0]

        vec_env = pfrl.envs.MultiprocessVectorEnv([make_env for _ in range(num_envs)])
        return vec_env, 1.0
