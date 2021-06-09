import logging
import os
import tempfile
import warnings

import numpy as np
import pytest
import torch
from torch import nn

import pfrl
from pfrl.agents import a3c
from pfrl.envs.abc import ABC
from pfrl.experiments.evaluator import run_evaluation_episodes
from pfrl.experiments.train_agent_async import train_agent_async
from pfrl.nn import RecurrentBranched, RecurrentSequential
from pfrl.policies import (
    GaussianHeadWithStateIndependentCovariance,
    SoftmaxCategoricalHead,
)


class _TestA3C:
    @pytest.fixture(autouse=True)
    def setUp(self):
        self.outdir = tempfile.mkdtemp()
        logging.basicConfig(level=logging.DEBUG)

    @pytest.mark.async_
    @pytest.mark.slow
    def test_abc(self):
        self._test_abc(
            self.t_max,
            recurrent=self.recurrent,
            episodic=self.episodic,
            discrete=self.discrete,
        )

    @pytest.mark.async_
    def test_abc_fast(self):
        self._test_abc(
            self.t_max,
            recurrent=self.recurrent,
            episodic=self.episodic,
            discrete=self.discrete,
            steps=10,
            require_success=False,
        )

    def make_model(self, env):
        hidden_size = 20
        obs_size = env.observation_space.low.size

        def weight_scale(layer, scale):
            with torch.no_grad():
                layer.weight.mul_(scale)
            return layer

        if self.recurrent:
            v = RecurrentSequential(
                nn.LSTM(num_layers=1, input_size=obs_size, hidden_size=hidden_size),
                weight_scale(nn.Linear(hidden_size, 1), 1e-1),
            )
            if self.discrete:
                n_actions = env.action_space.n
                pi = RecurrentSequential(
                    nn.LSTM(num_layers=1, input_size=obs_size, hidden_size=hidden_size),
                    weight_scale(nn.Linear(hidden_size, n_actions), 1e-1),
                    SoftmaxCategoricalHead(),
                )
            else:
                action_size = env.action_space.low.size
                pi = RecurrentSequential(
                    nn.LSTM(num_layers=1, input_size=obs_size, hidden_size=hidden_size),
                    weight_scale(nn.Linear(hidden_size, action_size), 1e-1),
                    GaussianHeadWithStateIndependentCovariance(
                        action_size=action_size,
                        var_type="diagonal",
                        var_func=lambda x: torch.exp(2 * x),
                        var_param_init=0,
                    ),
                )
            return RecurrentBranched(pi, v)
        else:
            v = nn.Sequential(
                nn.Linear(obs_size, hidden_size),
                nn.Tanh(),
                weight_scale(nn.Linear(hidden_size, 1), 1e-1),
            )
            if self.discrete:
                n_actions = env.action_space.n
                pi = nn.Sequential(
                    nn.Linear(obs_size, hidden_size),
                    nn.Tanh(),
                    weight_scale(nn.Linear(hidden_size, n_actions), 1e-1),
                    SoftmaxCategoricalHead(),
                )
            else:
                action_size = env.action_space.low.size
                pi = nn.Sequential(
                    nn.Linear(obs_size, hidden_size),
                    nn.Tanh(),
                    weight_scale(nn.Linear(hidden_size, action_size), 1e-1),
                    GaussianHeadWithStateIndependentCovariance(
                        action_size=action_size,
                        var_type="diagonal",
                        var_func=lambda x: torch.exp(2 * x),
                        var_param_init=0,
                    ),
                )
            return pfrl.nn.Branched(pi, v)

    def _test_abc(
        self,
        t_max,
        recurrent,
        discrete=True,
        episodic=True,
        steps=100000,
        require_success=True,
    ):

        nproc = 8

        def make_env(process_idx, test):
            size = 2
            return ABC(
                size=size,
                discrete=discrete,
                episodic=episodic or test,
                partially_observable=self.recurrent,
                deterministic=test,
            )

        env = make_env(0, False)

        model = self.make_model(env)

        from pfrl.optimizers import SharedRMSpropEpsInsideSqrt

        opt = SharedRMSpropEpsInsideSqrt(model.parameters())
        gamma = 0.8
        beta = 1e-2
        agent = a3c.A3C(
            model,
            opt,
            t_max=t_max,
            gamma=gamma,
            beta=beta,
            act_deterministically=True,
            max_grad_norm=1.0,
            recurrent=recurrent,
        )

        max_episode_len = None if episodic else 2

        with warnings.catch_warnings(record=True) as warns:
            train_agent_async(
                outdir=self.outdir,
                processes=nproc,
                make_env=make_env,
                agent=agent,
                steps=steps,
                max_episode_len=max_episode_len,
                eval_interval=500,
                eval_n_steps=None,
                eval_n_episodes=5,
                successful_score=1,
            )
            assert len(warns) == 0, warns[0]

        # The agent returned by train_agent_async is not guaranteed to be
        # successful because parameters could be modified by other processes
        # after success. Thus here the successful model is loaded explicitly.
        if require_success:
            agent.load(os.path.join(self.outdir, "successful"))

        # Test
        env = make_env(0, True)
        n_test_runs = 5
        eval_returns, _ = run_evaluation_episodes(
            env,
            agent,
            n_steps=None,
            n_episodes=n_test_runs,
            max_episode_len=max_episode_len,
        )
        successful_return = 1
        if require_success:
            n_succeeded = np.sum(np.asarray(eval_returns) >= successful_return)
            assert n_succeeded == n_test_runs


@pytest.mark.parametrize("t_max", [1, 2])
@pytest.mark.parametrize("recurrent", [False])
@pytest.mark.parametrize("discrete", [True, False])
@pytest.mark.parametrize("episodic", [True, False])
class TestA3CSmallTMax(_TestA3C):
    @pytest.fixture(autouse=True)
    def set_params(self, t_max, recurrent, discrete, episodic):
        self.t_max = t_max
        self.recurrent = recurrent
        self.discrete = discrete
        self.episodic = episodic


@pytest.mark.parametrize("t_max", [5])
@pytest.mark.parametrize("recurrent", [True, False])
@pytest.mark.parametrize("discrete", [True, False])
@pytest.mark.parametrize("episodic", [True, False])
class TestA3CLargeTMax(_TestA3C):
    @pytest.fixture(autouse=True)
    def set_params(self, t_max, recurrent, discrete, episodic):
        self.t_max = t_max
        self.recurrent = recurrent
        self.discrete = discrete
        self.episodic = episodic
