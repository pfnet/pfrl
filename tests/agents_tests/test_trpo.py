import os
import tempfile
import unittest

import numpy as np
import pytest
import torch
from torch import nn

import pfrl
from pfrl.agents import trpo
from pfrl.envs.abc import ABC
from pfrl.experiments import (
    train_agent_batch_with_evaluation,
    train_agent_with_evaluation,
)
from pfrl.experiments.evaluator import (
    batch_run_evaluation_episodes,
    run_evaluation_episodes,
)
from pfrl.nn import RecurrentSequential
from pfrl.policies import (
    GaussianHeadWithStateIndependentCovariance,
    SoftmaxCategoricalHead,
)
from pfrl.testing import torch_assert_allclose


def compute_hessian_vector_product(y, params, vec):
    grads = torch.autograd.grad([y], params, create_graph=True)
    flat_grads = trpo._flatten_and_concat_variables(grads)
    return trpo._hessian_vector_product(flat_grads, params, vec)


def compute_hessian(y, params):
    grads = torch.autograd.grad([y], params, create_graph=True)
    flat_grads = trpo._flatten_and_concat_variables(grads)
    hessian_rows = []
    for i in range(len(flat_grads)):
        ggrads = torch.autograd.grad([flat_grads[i]], params, retain_graph=True)
        assert all(ggrad is not None for ggrad in ggrads)
        flat_ggrads_data = trpo._flatten_and_concat_variables(ggrads).detach()
        hessian_rows.append(flat_ggrads_data)
    return torch.stack(hessian_rows)


class NonCudnnLSTM(nn.LSTM):
    """Non-cuDNN LSTM that supports double backprop.

    This is a workaround to address the issue that cuDNN RNNs in PyTorch
    do not support double backprop.

    See https://github.com/pytorch/pytorch/issues/5261.
    """

    def forward(self, x, recurrent_state):
        with torch.backends.cudnn.flags(enabled=False):
            return super().forward(x, recurrent_state)


class TestHessianVectorProduct(unittest.TestCase):
    def _generate_params_and_first_order_output(self):
        a = torch.rand(3, requires_grad=True)
        b = torch.rand(1, requires_grad=True)
        params = [a, b]
        y = torch.sum(a, dim=0, keepdims=True) * 3 + b
        return params, y

    def _generate_params_and_second_order_output(self):
        a = torch.rand(3, requires_grad=True)
        b = torch.rand(1, requires_grad=True)
        params = [a, b]
        y = torch.sum(a, dim=0, keepdims=True) * 3 * b
        return params, y

    def test_first_order(self):
        # First order, so its Hessian will contain None
        params, y = self._generate_params_and_first_order_output()

        vec = torch.rand(4)
        # Hessian-vector product computation should raise an error
        with self.assertRaises(RuntimeError):
            compute_hessian_vector_product(y, params, vec)

    def test_second_order(self):
        # Second order, so its Hessian will be non-zero
        params, y = self._generate_params_and_second_order_output()

        def test_hessian_vector_product_nonzero(vec):
            hvp = compute_hessian_vector_product(y, params, vec)
            hessian = compute_hessian(y, params)
            self.assertGreater(np.count_nonzero(hvp.numpy()), 0)
            self.assertGreater(np.count_nonzero(hessian.numpy()), 0)
            torch_assert_allclose(hvp, torch.matmul(hessian, vec), atol=1e-3)

        # Test with two different random vectors, reusing y
        test_hessian_vector_product_nonzero(torch.rand(4))
        test_hessian_vector_product_nonzero(torch.rand(4))


class _TestTRPO:
    @pytest.fixture(autouse=True)
    def setUp(self):
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
        n_test_runs = 10
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
        self, steps=100000, require_success=True, gpu=-1, load_model=False, num_envs=4
    ):

        env, _ = self.make_vec_env_and_successful_return(test=False, num_envs=num_envs)
        test_env, successful_return = self.make_vec_env_and_successful_return(
            test=True, num_envs=num_envs
        )
        agent = self.make_agent(env, gpu)
        max_episode_len = None if self.episodic else 2

        if load_model:
            print("Load agent from", self.agent_dirname)
            agent.load(self.agent_dirname)

        # Train
        train_agent_batch_with_evaluation(
            agent=agent,
            env=env,
            steps=steps,
            outdir=self.tmpdir,
            eval_interval=200,
            eval_n_steps=None,
            eval_n_episodes=40,
            successful_score=successful_return,
            eval_env=test_env,
            log_interval=100,
            max_episode_len=max_episode_len,
        )
        env.close()

        # Test
        n_test_runs = 10
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
            n_succeeded == n_test_runs

        # Save
        agent.save(self.agent_dirname)

    def make_agent(self, env, gpu):
        policy, vf = self.make_model(env)
        vf_opt = torch.optim.Adam(vf.parameters(), lr=1e-2)

        if self.standardize_obs:
            obs_normalizer = pfrl.nn.EmpiricalNormalization(
                env.observation_space.low.size
            )
        else:
            obs_normalizer = None

        agent = pfrl.agents.TRPO(
            policy=policy,
            vf=vf,
            vf_optimizer=vf_opt,
            obs_normalizer=obs_normalizer,
            gpu=gpu,
            gamma=0.5,
            lambd=self.lambd,
            entropy_coef=self.entropy_coef,
            standardize_advantages=self.standardize_advantages,
            update_interval=64,
            vf_batch_size=32,
            act_deterministically=True,
            recurrent=self.recurrent,
            max_grad_norm=1.0,
        )

        return agent

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
                    NonCudnnLSTM(
                        num_layers=1, input_size=obs_size, hidden_size=hidden_size
                    ),
                    weight_scale(nn.Linear(hidden_size, n_actions), 1e-1),
                    SoftmaxCategoricalHead(),
                )
            else:
                action_size = env.action_space.low.size
                pi = RecurrentSequential(
                    NonCudnnLSTM(
                        num_layers=1, input_size=obs_size, hidden_size=hidden_size
                    ),
                    weight_scale(nn.Linear(hidden_size, action_size), 1e-1),
                    GaussianHeadWithStateIndependentCovariance(
                        action_size=action_size,
                        var_type="diagonal",
                        var_func=lambda x: torch.exp(2 * x),
                        var_param_init=0,
                    ),
                )
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
        return pi, v

    def make_env_and_successful_return(self, test):
        env = ABC(
            discrete=self.discrete,
            episodic=self.episodic or test,
            deterministic=test,
            partially_observable=self.recurrent,
        )
        return env, 1

    def make_vec_env_and_successful_return(self, test, num_envs=3):
        def make_env():
            return self.make_env_and_successful_return(test)[0]

        vec_env = pfrl.envs.MultiprocessVectorEnv([make_env for _ in range(num_envs)])
        return vec_env, 1.0


@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("episodic", [False, True])
@pytest.mark.parametrize("lambd", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("entropy_coef", [0.0, 1e-5])
@pytest.mark.parametrize("standardize_advantages", [False, True])
@pytest.mark.parametrize("standardize_obs", [False, True])
class TestTRPONonRecurrent(_TestTRPO):
    @pytest.fixture(autouse=True)
    def set_params(
        self,
        discrete,
        episodic,
        lambd,
        entropy_coef,
        standardize_advantages,
        standardize_obs,
    ):
        self.discrete = discrete
        self.episodic = episodic
        self.lambd = lambd
        self.entropy_coef = entropy_coef
        self.standardize_advantages = standardize_advantages
        self.standardize_obs = standardize_obs
        self.recurrent = False


@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("episodic", [False, True])
@pytest.mark.parametrize("lambd", [0.9])
@pytest.mark.parametrize("entropy_coef", [1e-5])
@pytest.mark.parametrize("standardize_advantages", [True])
@pytest.mark.parametrize("standardize_obs", [True])
class TestTRPORecurrent(_TestTRPO):
    @pytest.fixture(autouse=True)
    def set_params(
        self,
        discrete,
        episodic,
        lambd,
        entropy_coef,
        standardize_advantages,
        standardize_obs,
    ):
        self.discrete = discrete
        self.episodic = episodic
        self.lambd = lambd
        self.entropy_coef = entropy_coef
        self.standardize_advantages = standardize_advantages
        self.standardize_obs = standardize_obs
        self.recurrent = True
