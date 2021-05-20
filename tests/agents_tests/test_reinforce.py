import logging
import tempfile

import numpy as np
import pytest
import torch
from torch import nn

import pfrl
from pfrl.envs.abc import ABC
from pfrl.experiments.evaluator import run_evaluation_episodes
from pfrl.policies import (
    GaussianHeadWithStateIndependentCovariance,
    SoftmaxCategoricalHead,
)


@pytest.mark.parametrize("discrete", [True, False])
@pytest.mark.parametrize("use_lstm", [True, False])
@pytest.mark.parametrize("batchsize", [1, 10])
@pytest.mark.parametrize("backward_separately", [True, False])
class TestREINFORCE:
    @pytest.fixture(autouse=True)
    def setUp(self, discrete, use_lstm, batchsize, backward_separately):
        self.discrete = discrete
        self.use_lstm = use_lstm
        self.batchsize = batchsize
        self.backward_separately = backward_separately
        self.outdir = tempfile.mkdtemp()
        logging.basicConfig(level=logging.DEBUG)

    @pytest.mark.slow
    def test_abc_cpu(self):
        self._test_abc(self.use_lstm, discrete=self.discrete)

    @pytest.mark.slow
    @pytest.mark.gpu
    def test_abc_gpu(self):
        self._test_abc(self.use_lstm, discrete=self.discrete, gpu=0)

    def test_abc_fast_cpu(self):
        self._test_abc(
            self.use_lstm, discrete=self.discrete, steps=10, require_success=False
        )

    @pytest.mark.gpu
    def test_abc_fast_gpu(self):
        self._test_abc(
            self.use_lstm,
            discrete=self.discrete,
            steps=10,
            require_success=False,
            gpu=0,
        )

    def _test_abc(
        self, use_lstm, discrete=True, steps=1000000, require_success=True, gpu=-1
    ):
        def make_env(process_idx, test):
            size = 2
            return ABC(
                size=size,
                discrete=discrete,
                episodic=True,
                partially_observable=self.use_lstm,
                deterministic=test,
            )

        sample_env = make_env(0, False)
        action_space = sample_env.action_space
        obs_space = sample_env.observation_space

        hidden_size = 20
        obs_size = obs_space.low.size
        if discrete:
            output_size = action_space.n
            head = SoftmaxCategoricalHead()
        else:
            output_size = action_space.low.size
            head = GaussianHeadWithStateIndependentCovariance(
                output_size, var_type="diagonal"
            )
        if use_lstm:
            model = pfrl.nn.RecurrentSequential(
                nn.LSTM(
                    num_layers=1,
                    input_size=obs_size,
                    hidden_size=hidden_size,
                ),
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, output_size),
                head,
            )
        else:
            model = nn.Sequential(
                nn.Linear(obs_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, output_size),
                head,
            )
        opt = torch.optim.Adam(model.parameters())
        beta = 1e-2
        agent = pfrl.agents.REINFORCE(
            model,
            opt,
            gpu=gpu,
            beta=beta,
            batchsize=self.batchsize,
            backward_separately=self.backward_separately,
            act_deterministically=True,
            recurrent=use_lstm,
        )

        pfrl.experiments.train_agent_with_evaluation(
            agent=agent,
            env=make_env(0, False),
            eval_env=make_env(0, True),
            outdir=self.outdir,
            steps=steps,
            train_max_episode_len=2,
            eval_interval=500,
            eval_n_steps=None,
            eval_n_episodes=5,
            successful_score=1,
        )

        # Test
        env = make_env(0, True)
        n_test_runs = 5
        eval_returns, _ = run_evaluation_episodes(
            env,
            agent,
            n_steps=None,
            n_episodes=n_test_runs,
        )
        if require_success:
            successful_return = 1
            n_succeeded = np.sum(np.asarray(eval_returns) >= successful_return)
            assert n_succeeded == n_test_runs
