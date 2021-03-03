import copy
import itertools
import os
import tempfile
import unittest

import numpy as np
import pytest
import torch
from torch import nn

import pfrl
from pfrl.agents import ppo
from pfrl.agents.ppo import PPO
from pfrl.envs.abc import ABC
from pfrl.experiments import (
    train_agent_batch_with_evaluation,
    train_agent_with_evaluation,
)
from pfrl.experiments.evaluator import (
    batch_run_evaluation_episodes,
    run_evaluation_episodes,
)
from pfrl.nn import RecurrentBranched, RecurrentSequential
from pfrl.policies import (
    GaussianHeadWithStateIndependentCovariance,
    SoftmaxCategoricalHead,
)
from pfrl.testing import torch_assert_allclose
from pfrl.utils.batch_states import batch_states


def make_random_episodes(n_episodes=10, obs_size=2, n_actions=3):
    episodes = []
    for _ in range(n_episodes):
        episode_length = np.random.randint(1, 100)
        episode = []
        last_state = np.random.uniform(-1, 1, size=obs_size)
        for t in range(episode_length):
            state = np.random.uniform(-1, 1, size=obs_size)
            episode.append(
                {
                    "state": last_state,
                    "action": np.random.randint(n_actions),
                    "reward": np.random.uniform(-1, 1),
                    "nonterminal": (
                        np.random.randint(2) if t == episode_length - 1 else 1
                    ),
                    "next_state": state,
                    "recurrent_state": None,
                    "next_recurrent_state": None,
                }
            )
            last_state = state
        episodes.append(episode)

    assert len(episodes) == n_episodes
    return episodes


class TestYieldSubsetOfSequencesWithFixedNumberOfItems(unittest.TestCase):
    def test_manual(self):
        episodes = [
            [1, 2, 3],
            [4, 5],
            [6, 7, 8],
            [9],
            [10, 11, 12],
        ]
        self.assertEqual(
            list(
                ppo._yield_subset_of_sequences_with_fixed_number_of_items(episodes, 4)
            ),
            [
                [[1, 2, 3], [4]],
                [[5], [6, 7, 8]],
                [[9], [10, 11, 12]],
            ],
        )
        self.assertEqual(
            list(
                ppo._yield_subset_of_sequences_with_fixed_number_of_items(episodes, 3)
            ),
            [
                [[1, 2, 3]],
                [[4, 5], [6]],
                [[7, 8], [9]],
                [[10, 11, 12]],
            ],
        )
        self.assertEqual(
            list(
                ppo._yield_subset_of_sequences_with_fixed_number_of_items(episodes, 2)
            ),
            [
                [[1, 2]],
                [[3], [4]],
                [[5], [6]],
                [[7, 8]],
                [[9], [10]],
                [[11, 12]],
            ],
        )


class TestLimitSequenceLength(unittest.TestCase):
    def test_manual(self):
        episodes = [
            [1, 2, 3],
            [4, 5],
            [6, 7, 8],
            [9],
        ]
        self.assertEqual(
            ppo._limit_sequence_length(episodes, 1),
            [[1], [2], [3], [4], [5], [6], [7], [8], [9]],
        )
        self.assertEqual(
            ppo._limit_sequence_length(episodes, 2),
            [
                [1, 2],
                [3],
                [4, 5],
                [6, 7],
                [8],
                [9],
            ],
        )
        self.assertEqual(
            ppo._limit_sequence_length(episodes, 3),
            episodes,
        )
        self.assertEqual(
            ppo._limit_sequence_length(episodes, 4),
            episodes,
        )

    def test_random(self):
        episodes = make_random_episodes()
        limit = 5
        new_episodes = pfrl.agents.ppo._limit_sequence_length(episodes, limit)
        for ep in new_episodes:
            self.assertLessEqual(len(ep), limit)
        # They should have the same number of transitions
        self.assertEqual(
            sum(len(ep) for ep in episodes), sum(len(ep) for ep in new_episodes)
        )


@pytest.mark.parametrize("use_obs_normalizer", [True, False])
@pytest.mark.parametrize("gamma", [1, 0.8, 0])
@pytest.mark.parametrize("lambd", [1, 0.8, 0])
@pytest.mark.parametrize("max_recurrent_sequence_len", [None, 7])
def test_ppo_dataset_recurrent_and_non_recurrent_equivalence(
    use_obs_normalizer, gamma, lambd, max_recurrent_sequence_len
):
    """Test equivalence between recurrent and non-recurrent datasets.

    When the same feed-forward model is used, the values of
    log_prob, v_pred, next_v_pred obtained by both recurrent and
    non-recurrent dataset creation functions should be the same.
    """
    episodes = make_random_episodes()
    if use_obs_normalizer:
        obs_normalizer = pfrl.nn.EmpiricalNormalization(2, clip_threshold=5)
        obs_normalizer.experience(torch.rand(10, 2))
    else:
        obs_normalizer = None

    def phi(obs):
        return (obs * 0.5).astype(np.float32)

    device = torch.device("cpu")

    obs_size = 2
    n_actions = 3

    non_recurrent_model = pfrl.nn.Branched(
        nn.Sequential(
            nn.Linear(obs_size, n_actions),
            SoftmaxCategoricalHead(),
        ),
        nn.Linear(obs_size, 1),
    )
    recurrent_model = RecurrentSequential(
        non_recurrent_model,
    )

    dataset = pfrl.agents.ppo._make_dataset(
        episodes=copy.deepcopy(episodes),
        model=non_recurrent_model,
        phi=phi,
        batch_states=batch_states,
        obs_normalizer=obs_normalizer,
        gamma=gamma,
        lambd=lambd,
        device=device,
    )

    dataset_recurrent = pfrl.agents.ppo._make_dataset_recurrent(
        episodes=copy.deepcopy(episodes),
        model=recurrent_model,
        phi=phi,
        batch_states=batch_states,
        obs_normalizer=obs_normalizer,
        gamma=gamma,
        lambd=lambd,
        max_recurrent_sequence_len=max_recurrent_sequence_len,
        device=device,
    )

    assert "log_prob" not in episodes[0][0]
    assert "log_prob" in dataset[0]
    assert "log_prob" in dataset_recurrent[0][0]
    # They are not just shallow copies
    assert dataset[0]["log_prob"] is not dataset_recurrent[0][0]["log_prob"]

    states = [tr["state"] for tr in dataset]
    recurrent_states = [
        tr["state"] for tr in itertools.chain.from_iterable(dataset_recurrent)
    ]
    torch_assert_allclose(states, recurrent_states)

    actions = [tr["action"] for tr in dataset]
    recurrent_actions = [
        tr["action"] for tr in itertools.chain.from_iterable(dataset_recurrent)
    ]
    torch_assert_allclose(actions, recurrent_actions)

    rewards = [tr["reward"] for tr in dataset]
    recurrent_rewards = [
        tr["reward"] for tr in itertools.chain.from_iterable(dataset_recurrent)
    ]
    torch_assert_allclose(rewards, recurrent_rewards)

    nonterminals = [tr["nonterminal"] for tr in dataset]
    recurrent_nonterminals = [
        tr["nonterminal"] for tr in itertools.chain.from_iterable(dataset_recurrent)
    ]
    torch_assert_allclose(nonterminals, recurrent_nonterminals)

    log_probs = [tr["log_prob"] for tr in dataset]
    recurrent_log_probs = [
        tr["log_prob"] for tr in itertools.chain.from_iterable(dataset_recurrent)
    ]
    torch_assert_allclose(log_probs, recurrent_log_probs)

    vs_pred = [tr["v_pred"] for tr in dataset]
    recurrent_vs_pred = [
        tr["v_pred"] for tr in itertools.chain.from_iterable(dataset_recurrent)
    ]
    torch_assert_allclose(vs_pred, recurrent_vs_pred)

    next_vs_pred = [tr["next_v_pred"] for tr in dataset]
    recurrent_next_vs_pred = [
        tr["next_v_pred"] for tr in itertools.chain.from_iterable(dataset_recurrent)
    ]
    torch_assert_allclose(next_vs_pred, recurrent_next_vs_pred)

    advs = [tr["adv"] for tr in dataset]
    recurrent_advs = [
        tr["adv"] for tr in itertools.chain.from_iterable(dataset_recurrent)
    ]
    torch_assert_allclose(advs, recurrent_advs)

    vs_teacher = [tr["v_teacher"] for tr in dataset]
    recurrent_vs_teacher = [
        tr["v_teacher"] for tr in itertools.chain.from_iterable(dataset_recurrent)
    ]
    torch_assert_allclose(vs_teacher, recurrent_vs_teacher)


class _TestPPO:
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
        max_episode_len = None if self.episodic else 2

        if load_model:
            print("Load agent from", self.agent_dirname)
            agent.load(self.agent_dirname)

        # Train
        train_agent_with_evaluation(
            agent=agent,
            env=env,
            steps=steps,
            outdir=self.tmpdir,
            eval_interval=200,
            eval_n_steps=None,
            eval_n_episodes=50,
            successful_score=successful_return,
            eval_env=test_env,
            train_max_episode_len=max_episode_len,
        )

        # Test
        n_test_runs = 10
        eval_returns = run_evaluation_episodes(
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
        eval_returns = batch_run_evaluation_episodes(
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
        model = self.make_model(env)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        return self.make_ppo_agent(env=env, model=model, opt=opt, gpu=gpu)

    def make_ppo_agent(self, env, model, opt, gpu):
        return PPO(
            model,
            opt,
            gpu=gpu,
            gamma=0.8,
            lambd=self.lambd,
            update_interval=64,
            minibatch_size=16,
            epochs=3,
            clip_eps_vf=self.clip_eps_vf,
            standardize_advantages=self.standardize_advantages,
            recurrent=self.recurrent,
            entropy_coef=1e-5,
            act_deterministically=True,
            max_grad_norm=1.0,
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

    def make_env_and_successful_return(self, test):
        env = ABC(
            discrete=self.discrete,
            deterministic=test,
            episodic=self.episodic,
            partially_observable=self.recurrent,
        )
        return env, 1.0

    def make_vec_env_and_successful_return(self, test, num_envs=3):
        def make_env():
            return self.make_env_and_successful_return(test)[0]

        vec_env = pfrl.envs.MultiprocessVectorEnv([make_env for _ in range(num_envs)])
        return vec_env, 1.0


@pytest.mark.parametrize("clip_eps_vf", [None, 0.2])
@pytest.mark.parametrize("lambd", [0.0, 0.5])
@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("standardize_advantages", [False, True])
@pytest.mark.parametrize("episodic", [True, False])
class TestPPONonRecurrent(_TestPPO):
    @pytest.fixture(autouse=True)
    def set_params(
        self,
        clip_eps_vf,
        lambd,
        discrete,
        standardize_advantages,
        episodic,
    ):
        self.clip_eps_vf = clip_eps_vf
        self.lambd = lambd
        self.discrete = discrete
        self.standardize_advantages = standardize_advantages
        self.episodic = episodic
        self.recurrent = False


@pytest.mark.parametrize("clip_eps_vf", [0.2])
@pytest.mark.parametrize("lambd", [0.0, 0.5])
@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("standardize_advantages", [True])
@pytest.mark.parametrize("episodic", [True, False])
class TestPPORecurrent(_TestPPO):
    @pytest.fixture(autouse=True)
    def set_params(
        self,
        clip_eps_vf,
        lambd,
        discrete,
        standardize_advantages,
        episodic,
    ):
        self.clip_eps_vf = clip_eps_vf
        self.lambd = lambd
        self.discrete = discrete
        self.standardize_advantages = standardize_advantages
        self.episodic = episodic
        self.recurrent = True


def test_yield_minibatches_divisible():
    dataset = [1, 2, 3, 4]
    minibatches = list(ppo._yield_minibatches(dataset, minibatch_size=2, num_epochs=3))
    assert len(minibatches) == 6
    samples = sum(minibatches, [])
    assert len(samples) == 12
    assert {1, 2, 3, 4} == set(samples[:4])
    assert {1, 2, 3, 4} == set(samples[4:8])
    assert {1, 2, 3, 4} == set(samples[8:12])


def test_yield_minibatches_indivisible():
    dataset = [1, 2, 3]
    minibatches = list(ppo._yield_minibatches(dataset, minibatch_size=2, num_epochs=3))
    assert len(minibatches) == 5
    samples = sum(minibatches, [])
    assert len(samples) == 10
    # samples[:6] is from the first two epochs
    assert samples[:6].count(1) == 2
    assert samples[:6].count(2) == 2
    assert samples[:6].count(3) == 2
    # samples[6:] is from the final epoch
    assert 1 <= samples[6:].count(1) <= 2
    assert 1 <= samples[6:].count(2) <= 2
    assert 1 <= samples[6:].count(3) <= 2


def test_yield_minibatches_smaller_dataset():
    # dataset smaller than minibatch
    dataset = [1, 2]
    minibatches = list(ppo._yield_minibatches(dataset, minibatch_size=4, num_epochs=3))
    assert len(minibatches) == 2
    samples = sum(minibatches, [])
    assert len(samples) == 8
    assert samples.count(1) == 4
    assert samples.count(2) == 4
