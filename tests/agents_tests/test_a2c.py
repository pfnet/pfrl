import os
import tempfile

import numpy as np
import pytest
import torch
from torch import nn

import pfrl
from pfrl.agents.a2c import A2C
from pfrl.envs.abc import ABC
from pfrl.experiments.evaluator import batch_run_evaluation_episodes
from pfrl.nn import Branched
from pfrl.policies import (
    GaussianHeadWithStateIndependentCovariance,
    SoftmaxCategoricalHead,
)


@pytest.mark.parametrize("num_processes", [1, 3])
@pytest.mark.parametrize("use_gae", [False, True])
@pytest.mark.parametrize("discrete", [False, True])
class TestA2C:
    @pytest.fixture(autouse=True)
    def setUp(self, num_processes, use_gae, discrete):
        self.num_processes = num_processes
        self.use_gae = use_gae
        self.discrete = discrete
        self.tmpdir = tempfile.mkdtemp()
        self.agent_dirname = os.path.join(self.tmpdir, "agent_final")

    @pytest.mark.slow
    def test_abc_cpu(self):
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

    def _test_abc(self, steps=1000000, require_success=True, gpu=-1, load_model=False):

        env, _ = self.make_env_and_successful_return(test=False, n=self.num_processes)
        test_env, successful_return = self.make_env_and_successful_return(
            test=True, n=1
        )
        agent = self.make_agent(env, gpu)

        if load_model:
            print("Load agent from", self.agent_dirname)
            agent.load(self.agent_dirname)

        # Train
        pfrl.experiments.train_agent_batch_with_evaluation(
            agent=agent,
            env=env,
            steps=steps,
            outdir=self.tmpdir,
            log_interval=10,
            eval_interval=200,
            eval_n_steps=None,
            eval_n_episodes=50,
            successful_score=1,
            eval_env=test_env,
        )
        env.close()

        # Test
        n_test_runs = 100
        eval_returns, _ = batch_run_evaluation_episodes(
            test_env,
            agent,
            n_steps=None,
            n_episodes=n_test_runs,
        )
        test_env.close()
        n_succeeded = np.sum(np.asarray(eval_returns) >= successful_return)
        if require_success:
            assert n_succeeded > 0.8 * n_test_runs

        # Save
        agent.save(self.agent_dirname)

    def make_agent(self, env, gpu):
        model = self.make_model(env)
        opt = torch.optim.Adam(model.parameters(), lr=3e-4)
        return self.make_a2c_agent(
            env=env, model=model, opt=opt, gpu=gpu, num_processes=self.num_processes
        )

    def make_a2c_agent(self, env, model, opt, gpu, num_processes):
        return A2C(
            model,
            opt,
            gpu=gpu,
            gamma=0.99,
            num_processes=num_processes,
            use_gae=self.use_gae,
        )

    def make_model(self, env):
        hidden_size = 50

        obs_size = env.observation_space.low.size
        v = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        if self.discrete:
            n_actions = env.action_space.n
            pi = nn.Sequential(
                nn.Linear(obs_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, n_actions),
                SoftmaxCategoricalHead(),
            )
        else:
            action_size = env.action_space.low.size
            pi = nn.Sequential(
                nn.Linear(obs_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, action_size),
                GaussianHeadWithStateIndependentCovariance(
                    action_size=action_size,
                    var_type="diagonal",
                    var_func=lambda x: torch.exp(2 * x),
                    var_param_init=0,
                ),
            )

        return Branched(pi, v)

    def make_env_and_successful_return(self, test, n):
        def make_env():
            return ABC(discrete=self.discrete, deterministic=test)

        vec_env = pfrl.envs.MultiprocessVectorEnv([make_env for _ in range(n)])
        return vec_env, 1
