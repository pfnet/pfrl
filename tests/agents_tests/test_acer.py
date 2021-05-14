import copy
import logging
import os
import tempfile
import warnings

import numpy as np
import pytest
import torch
from torch import nn

import pfrl
from pfrl.agents import acer
from pfrl.envs.abc import ABC
from pfrl.experiments.evaluator import run_evaluation_episodes
from pfrl.experiments.train_agent_async import train_agent_async
from pfrl.nn import ConcatObsAndAction
from pfrl.policies import GaussianHeadWithDiagonalCovariance, SoftmaxCategoricalHead
from pfrl.q_functions import DiscreteActionValueHead
from pfrl.replay_buffers import EpisodicReplayBuffer


def extract_gradients_as_single_vector(mod):
    return torch.cat([p.grad.flatten() for p in mod.parameters()]).numpy()


@pytest.mark.parametrize("distrib_type", ["Gaussian", "Softmax"])
@pytest.mark.parametrize("pi_deg", [True, False])
@pytest.mark.parametrize("mu_deg", [True, False])
@pytest.mark.parametrize("truncation_threshold", [0, 1, 10, None])
class TestDegenerateDistribution:
    @pytest.fixture(autouse=True)
    def setUp(self, distrib_type, pi_deg, mu_deg, truncation_threshold):
        self.distrib_type = distrib_type
        self.pi_deg = pi_deg
        self.mu_deg = mu_deg
        self.truncation_threshold = truncation_threshold
        if self.distrib_type == "Gaussian":
            action_size = 2
            W = torch.rand(1, action_size)
            self.action_value = pfrl.action_value.SingleActionValue(
                evaluator=lambda x: torch.nn.functional.linear(x, W)
            )
            nondeg_distrib = torch.distributions.Independent(
                torch.distributions.Normal(
                    loc=torch.rand(1, action_size),
                    scale=torch.full((1, action_size), 1),
                ),
                1,
            )
            deg_distrib = torch.distributions.Independent(
                torch.distributions.Normal(
                    loc=torch.rand(1, action_size),
                    scale=torch.full((1, action_size), 1e-10),
                ),
                1,
            )
        elif self.distrib_type == "Softmax":
            q_values = torch.as_tensor([[1, 3]], dtype=torch.float)
            self.action_value = pfrl.action_value.DiscreteActionValue(q_values)
            nondeg_logits = torch.as_tensor([[0, 0]], dtype=torch.float)
            nondeg_distrib = torch.distributions.Categorical(logits=nondeg_logits)
            deg_logits = torch.as_tensor([[1e10, 1e-10]], dtype=torch.float)
            deg_distrib = torch.distributions.Categorical(deg_logits)
        self.pi = deg_distrib if self.pi_deg else nondeg_distrib
        self.mu = deg_distrib if self.mu_deg else nondeg_distrib

    def test_importance(self):
        action = self.mu.sample()
        pimu = acer.compute_importance(self.pi, self.mu, action)
        print("pi/mu", pimu)
        assert np.isscalar(pimu)
        assert not np.isnan(pimu)

    def test_full_importance(self):
        if self.distrib_type == "Gaussian":
            pytest.skip()
        pimu = acer.compute_full_importance(self.pi, self.mu)
        print("pi/mu", pimu)
        assert isinstance(pimu, torch.Tensor)
        assert not np.isnan(pimu.numpy()).any()

    def test_full_correction_term(self):
        if self.distrib_type == "Gaussian":
            pytest.skip()
        if self.truncation_threshold is None:
            pytest.skip()
        correction_term = acer.compute_policy_gradient_full_correction(
            self.pi, self.mu, self.action_value, 0, self.truncation_threshold
        )
        print("correction_term", correction_term)
        assert isinstance(correction_term, torch.Tensor)
        assert correction_term.numel() == 1
        assert not np.isnan(float(correction_term))

    def test_sample_correction_term(self):
        if self.truncation_threshold is None:
            return
        correction_term = acer.compute_policy_gradient_sample_correction(
            self.pi, self.mu, self.action_value, 0, self.truncation_threshold
        )
        print("correction_term", correction_term)
        assert isinstance(correction_term, torch.Tensor)
        assert correction_term.numel() == 1
        assert not np.isnan(float(correction_term))

    def test_policy_gradient(self):
        action = self.mu.sample()
        pg = acer.compute_policy_gradient_loss(
            action, 1, self.pi, self.mu, self.action_value, 0, self.truncation_threshold
        )
        assert isinstance(pg, torch.Tensor)
        print("pg", pg)
        assert not np.isnan(pg.numpy()).any()


@pytest.mark.parametrize("action_size", [1, 2])
def test_bias_correction_gaussian(action_size):
    base_policy = nn.Sequential(
        nn.Linear(1, action_size * 2),
        GaussianHeadWithDiagonalCovariance(),
    )
    another_policy = nn.Sequential(
        nn.Linear(1, action_size * 2),
        GaussianHeadWithDiagonalCovariance(),
    )
    W = torch.rand(1, action_size)
    action_value = pfrl.action_value.SingleActionValue(
        evaluator=lambda x: torch.nn.functional.linear(x, W)
    )
    _test_bias_correction(base_policy, another_policy, action_value)


@pytest.mark.parametrize("n_actions", [2, 3])
def test_bias_correction_softmax(n_actions):
    base_policy = nn.Sequential(
        nn.Linear(1, n_actions),
        SoftmaxCategoricalHead(),
    )
    another_policy = nn.Sequential(
        nn.Linear(1, n_actions),
        SoftmaxCategoricalHead(),
    )
    q_values = torch.rand(1, n_actions)
    action_value = pfrl.action_value.DiscreteActionValue(q_values)
    _test_bias_correction(base_policy, another_policy, action_value)


def _test_bias_correction(base_policy, another_policy, action_value):
    x = torch.full((1, 1), 1, dtype=torch.float)
    pi = base_policy(x)
    mu = another_policy(x)

    def evaluate_action(action):
        return float(action_value.evaluate_actions(action))

    n = 1000

    pi_samples = [pi.sample() for _ in range(n)]
    mu_samples = [mu.sample() for _ in range(n)]

    onpolicy_gs = []
    for sample in pi_samples:
        base_policy.zero_grad()
        loss = -evaluate_action(sample) * pi.log_prob(sample)
        loss.squeeze().backward(retain_graph=True)
        onpolicy_gs.append(extract_gradients_as_single_vector(base_policy))
    # on-policy
    onpolicy_gs_mean = np.mean(onpolicy_gs, axis=0)
    onpolicy_gs_var = np.var(onpolicy_gs, axis=0)
    print("on-policy")
    print("g mean", onpolicy_gs_mean)
    print("g var", onpolicy_gs_var)

    # off-policy without importance sampling
    offpolicy_gs = []
    for sample in mu_samples:
        base_policy.zero_grad()
        loss = -evaluate_action(sample) * pi.log_prob(sample)
        loss.squeeze().backward(retain_graph=True)
        offpolicy_gs.append(extract_gradients_as_single_vector(base_policy))
    offpolicy_gs_mean = np.mean(offpolicy_gs, axis=0)
    offpolicy_gs_var = np.var(offpolicy_gs, axis=0)
    print("off-policy")
    print("g mean", offpolicy_gs_mean)
    print("g var", offpolicy_gs_var)

    # off-policy with importance sampling
    is_gs = []
    for sample in mu_samples:
        base_policy.zero_grad()
        rho = float((pi.log_prob(sample) - mu.log_prob(sample)).exp())
        loss = -rho * evaluate_action(sample) * pi.log_prob(sample)
        loss.squeeze().backward(retain_graph=True)
        is_gs.append(extract_gradients_as_single_vector(base_policy))
    is_gs_mean = np.mean(is_gs, axis=0)
    is_gs_var = np.var(is_gs, axis=0)
    print("importance sampling")
    print("g mean", is_gs_mean)
    print("g var", is_gs_var)

    # off-policy with truncated importance sampling + bias correction
    def bias_correction_policy_gradients(truncation_threshold):
        gs = []
        for sample in mu_samples:
            base_policy.zero_grad()
            loss = acer.compute_policy_gradient_loss(
                action=sample,
                advantage=evaluate_action(sample),
                action_distrib=pi,
                action_distrib_mu=mu,
                action_value=action_value,
                v=0,
                truncation_threshold=truncation_threshold,
            )
            loss.squeeze().backward(retain_graph=True)
            gs.append(extract_gradients_as_single_vector(base_policy))
        return gs

    # c=0 means on-policy sampling
    print("truncated importance sampling + bias correction c=0")
    tis_c0_gs = bias_correction_policy_gradients(0)
    tis_c0_gs_mean = np.mean(tis_c0_gs, axis=0)
    tis_c0_gs_var = np.var(tis_c0_gs, axis=0)
    print("g mean", tis_c0_gs_mean)
    print("g var", tis_c0_gs_var)
    # c=0 must be low-bias compared to naive off-policy sampling
    assert np.linalg.norm(onpolicy_gs_mean - tis_c0_gs_mean) <= np.linalg.norm(
        onpolicy_gs_mean - offpolicy_gs_mean
    )

    # c=1 means truncated importance sampling with bias correction
    print("truncated importance sampling + bias correction c=1")
    tis_c1_gs = bias_correction_policy_gradients(1)
    tis_c1_gs_mean = np.mean(tis_c1_gs, axis=0)
    tis_c1_gs_var = np.var(tis_c1_gs, axis=0)
    print("g mean", tis_c1_gs_mean)
    print("g var", tis_c1_gs_var)
    # c=1 must be low-variance compared to naive importance sampling
    assert tis_c1_gs_var.sum() <= is_gs_var.sum()
    # c=1 must be low-bias compared to naive off-policy sampling
    assert np.linalg.norm(onpolicy_gs_mean - tis_c1_gs_mean) < np.linalg.norm(
        onpolicy_gs_mean - offpolicy_gs_mean
    )

    # c=inf means importance sampling no truncation
    print("truncated importance sampling + bias correction c=inf")
    tis_cinf_gs = bias_correction_policy_gradients(np.inf)
    tis_cinf_gs_mean = np.mean(tis_cinf_gs, axis=0)
    tis_cinf_gs_var = np.var(tis_cinf_gs, axis=0)
    print("g mean", tis_cinf_gs_mean)
    print("g var", tis_cinf_gs_var)
    np.testing.assert_allclose(tis_cinf_gs_mean, is_gs_mean, rtol=1e-3)
    np.testing.assert_allclose(tis_cinf_gs_var, is_gs_var, rtol=1e-3)


def test_compute_loss_with_kl_constraint_gaussian():
    action_size = 3
    policy = nn.Sequential(
        nn.Linear(1, action_size * 2),
        GaussianHeadWithDiagonalCovariance(),
    )
    _test_compute_loss_with_kl_constraint(policy)


def test_compute_loss_with_kl_constraint_softmax():
    n_actions = 3
    policy = nn.Sequential(
        nn.Linear(1, n_actions),
        SoftmaxCategoricalHead(),
    )
    _test_compute_loss_with_kl_constraint(policy)


def _test_compute_loss_with_kl_constraint(base_policy):

    # Train a policy with and without KL constraint against the original
    # distribution to confirm KL constraint works.

    x = torch.rand(1, 1)

    with torch.no_grad():
        # Compute KL divergence against the original distribution
        base_distrib = base_policy(x)

    def base_loss_func(distrib):
        # Any loss that tends to increase KL divergence should be ok
        kl = torch.distributions.kl_divergence(base_distrib, distrib)
        return -(kl + distrib.entropy())

    def compute_kl_after_update(loss_func, n=100):
        policy = copy.deepcopy(base_policy)
        optimizer = torch.optim.SGD(policy.parameters(), 1e-2)
        for _ in range(n):
            distrib = policy(x)
            policy.zero_grad()
            loss_func(distrib).backward(retain_graph=True)
            optimizer.step()
        with torch.no_grad():
            distrib_after = policy(x)
            return float(torch.distributions.kl_divergence(base_distrib, distrib_after))

    # Without kl constraint
    kl_after_without_constraint = compute_kl_after_update(base_loss_func)
    print("kl_after_without_constraint", kl_after_without_constraint)

    # With kl constraint
    def loss_func_with_constraint(distrib):
        loss, kl = acer.compute_loss_with_kl_constraint(
            distrib, base_distrib, base_loss_func(distrib), delta=0
        )
        return loss

    kl_after_with_constraint = compute_kl_after_update(loss_func_with_constraint)
    print("kl_after_with_constraint", kl_after_with_constraint)

    # KL constraint should make KL divergence small after updates
    assert kl_after_with_constraint < kl_after_without_constraint


class _TestACER:
    @pytest.fixture(autouse=True)
    def setUp(self):
        self.outdir = tempfile.mkdtemp()
        logging.basicConfig(level=logging.DEBUG)

    @pytest.mark.async_
    @pytest.mark.slow
    def test_abc(self):
        self._test_abc(
            self.t_max, self.use_lstm, discrete=self.discrete, episodic=self.episodic
        )

    @pytest.mark.async_
    def test_abc_fast(self):
        self._test_abc(
            self.t_max,
            self.use_lstm,
            discrete=self.discrete,
            episodic=self.episodic,
            steps=10,
            require_success=False,
        )

    def _test_abc(
        self,
        t_max,
        use_lstm,
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
                partially_observable=self.use_lstm,
                deterministic=test,
            )

        sample_env = make_env(0, False)
        action_space = sample_env.action_space
        obs_space = sample_env.observation_space

        replay_buffer = EpisodicReplayBuffer(10 ** 4)
        obs_size = obs_space.low.size
        hidden_size = 20
        if discrete:
            n_actions = action_space.n
            head = acer.ACERDiscreteActionHead(
                pi=nn.Sequential(
                    nn.Linear(hidden_size, n_actions),
                    SoftmaxCategoricalHead(),
                ),
                q=nn.Sequential(
                    nn.Linear(hidden_size, n_actions),
                    DiscreteActionValueHead(),
                ),
            )
        else:
            action_size = action_space.low.size
            head = acer.ACERContinuousActionHead(
                pi=nn.Sequential(
                    nn.Linear(hidden_size, action_size * 2),
                    GaussianHeadWithDiagonalCovariance(),
                ),
                v=nn.Sequential(
                    nn.Linear(hidden_size, 1),
                ),
                adv=nn.Sequential(
                    ConcatObsAndAction(),
                    nn.Linear(hidden_size + action_size, 1),
                ),
            )
        if use_lstm:
            model = pfrl.nn.RecurrentSequential(
                nn.Linear(obs_size, hidden_size),
                nn.LeakyReLU(),
                nn.LSTM(num_layers=1, input_size=hidden_size, hidden_size=hidden_size),
                head,
            )
        else:
            model = nn.Sequential(
                nn.Linear(obs_size, hidden_size),
                nn.LeakyReLU(),
                head,
            )
        eps = 1e-8
        opt = pfrl.optimizers.SharedRMSpropEpsInsideSqrt(
            model.parameters(), lr=1e-3, eps=eps, alpha=0.99
        )
        gamma = 0.5
        beta = 1e-5
        if self.n_times_replay == 0 and self.disable_online_update:
            # At least one of them must be enabled
            pytest.skip()
        agent = acer.ACER(
            model,
            opt,
            replay_buffer=replay_buffer,
            t_max=t_max,
            gamma=gamma,
            beta=beta,
            n_times_replay=self.n_times_replay,
            act_deterministically=True,
            disable_online_update=self.disable_online_update,
            replay_start_size=100,
            use_trust_region=self.use_trust_region,
            recurrent=use_lstm,
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


@pytest.mark.parametrize("discrete", [True, False])
@pytest.mark.parametrize("t_max", [1, 2])
@pytest.mark.parametrize("episodic", [True, False])
@pytest.mark.parametrize("n_times_replay", [0, 2])
@pytest.mark.parametrize("disable_online_update", [True, False])
@pytest.mark.parametrize("use_trust_region", [True, False])
class TestACERNonRecurrent(_TestACER):
    @pytest.fixture(autouse=True)
    def set_params(
        self,
        discrete,
        t_max,
        episodic,
        n_times_replay,
        disable_online_update,
        use_trust_region,
    ):
        self.use_lstm = False
        self.discrete = discrete
        self.t_max = t_max
        self.episodic = episodic
        self.n_times_replay = n_times_replay
        self.disable_online_update = disable_online_update
        self.use_trust_region = use_trust_region


@pytest.mark.parametrize("discrete", [True, False])
@pytest.mark.parametrize("t_max", [5])
@pytest.mark.parametrize("episodic", [True, False])
@pytest.mark.parametrize("n_times_replay", [0, 2])
@pytest.mark.parametrize("disable_online_update", [True, False])
@pytest.mark.parametrize("use_trust_region", [True, False])
class TestACERRecurrent(_TestACER):
    @pytest.fixture(autouse=True)
    def set_params(
        self,
        discrete,
        t_max,
        episodic,
        n_times_replay,
        disable_online_update,
        use_trust_region,
    ):
        self.use_lstm = True
        self.discrete = discrete
        self.t_max = t_max
        self.episodic = episodic
        self.n_times_replay = n_times_replay
        self.disable_online_update = disable_online_update
        self.use_trust_region = use_trust_region
