import os

import numpy as np
import pytest
import torch
from torch import nn

import pfrl
import pfrl.nn as pnn
from pfrl import agents, explorers, replay_buffers
from pfrl.initializers import init_chainer_default
from pfrl.utils import download_model


@pytest.mark.parametrize("pretrained_type", ["final", "best"])
class TestLoadDQN:
    @pytest.fixture(autouse=True)
    def setup(self, pretrained_type):
        self.pretrained_type = pretrained_type

    def _test_load_dqn(self, gpu):
        from pfrl.q_functions import DiscreteActionValueHead

        n_actions = 4
        q_func = nn.Sequential(
            pnn.LargeAtariCNN(),
            init_chainer_default(nn.Linear(512, n_actions)),
            DiscreteActionValueHead(),
        )

        # Use the same hyperparameters as the Nature paper

        opt = pfrl.optimizers.RMSpropEpsInsideSqrt(
            q_func.parameters(),
            lr=2.5e-4,
            alpha=0.95,
            momentum=0.0,
            eps=1e-2,
            centered=True,
        )

        rbuf = replay_buffers.ReplayBuffer(100)

        explorer = explorers.LinearDecayEpsilonGreedy(
            start_epsilon=1.0,
            end_epsilon=0.1,
            decay_steps=10 ** 6,
            random_action_func=lambda: np.random.randint(4),
        )

        agent = agents.DQN(
            q_func,
            opt,
            rbuf,
            gpu=gpu,
            gamma=0.99,
            explorer=explorer,
            replay_start_size=50,
            target_update_interval=10 ** 4,
            clip_delta=True,
            update_interval=4,
            batch_accumulator="sum",
            phi=lambda x: x,
        )

        downloaded_model, exists = download_model(
            "DQN", "BreakoutNoFrameskip-v4", model_type=self.pretrained_type
        )
        agent.load(downloaded_model)
        if os.environ.get("PFRL_ASSERT_DOWNLOADED_MODEL_IS_CACHED"):
            assert exists

    def test_cpu(self):
        self._test_load_dqn(gpu=None)

    @pytest.mark.gpu
    def test_gpu(self):
        self._test_load_dqn(gpu=0)


@pytest.mark.parametrize("pretrained_type", ["final", "best"])
class TestLoadIQN:
    @pytest.fixture(autouse=True)
    def setup(self, pretrained_type):
        self.pretrained_type = pretrained_type

    def _test_load_iqn(self, gpu):
        n_actions = 4
        q_func = pfrl.agents.iqn.ImplicitQuantileQFunction(
            psi=nn.Sequential(
                nn.Conv2d(4, 32, 8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            ),
            phi=nn.Sequential(
                pfrl.agents.iqn.CosineBasisLinear(64, 3136),
                nn.ReLU(),
            ),
            f=nn.Sequential(
                nn.Linear(3136, 512),
                nn.ReLU(),
                nn.Linear(512, n_actions),
            ),
        )

        # Use the same hyper parameters as https://arxiv.org/abs/1710.10044
        opt = torch.optim.Adam(q_func.parameters(), lr=5e-5, eps=1e-2 / 32)

        rbuf = replay_buffers.ReplayBuffer(100)

        explorer = explorers.LinearDecayEpsilonGreedy(
            start_epsilon=1.0,
            end_epsilon=0.1,
            decay_steps=10 ** 6,
            random_action_func=lambda: np.random.randint(4),
        )

        agent = agents.IQN(
            q_func,
            opt,
            rbuf,
            gpu=gpu,
            gamma=0.99,
            explorer=explorer,
            replay_start_size=50,
            target_update_interval=10 ** 4,
            update_interval=4,
            batch_accumulator="mean",
            phi=lambda x: x,
            quantile_thresholds_N=64,
            quantile_thresholds_N_prime=64,
            quantile_thresholds_K=32,
        )

        downloaded_model, exists = download_model(
            "IQN", "BreakoutNoFrameskip-v4", model_type=self.pretrained_type
        )
        agent.load(downloaded_model)
        if os.environ.get("PFRL_ASSERT_DOWNLOADED_MODEL_IS_CACHED"):
            assert exists

    def test_cpu(self):
        self._test_load_iqn(gpu=None)

    @pytest.mark.gpu
    def test_gpu(self):
        self._test_load_iqn(gpu=0)


@pytest.mark.parametrize("pretrained_type", ["final", "best"])
class TestLoadRainbow:
    @pytest.fixture(autouse=True)
    def setup(self, pretrained_type):
        self.pretrained_type = pretrained_type

    def _test_load_rainbow(self, gpu):
        from pfrl.q_functions import DistributionalDuelingDQN

        q_func = DistributionalDuelingDQN(4, 51, -10, 10)
        pnn.to_factorized_noisy(q_func, sigma_scale=0.5)
        explorer = explorers.Greedy()
        opt = torch.optim.Adam(q_func.parameters(), 6.25e-5, eps=1.5 * 10 ** -4)
        rbuf = replay_buffers.ReplayBuffer(100)
        agent = agents.CategoricalDoubleDQN(
            q_func,
            opt,
            rbuf,
            gpu=gpu,
            gamma=0.99,
            explorer=explorer,
            minibatch_size=32,
            replay_start_size=50,
            target_update_interval=32000,
            update_interval=4,
            batch_accumulator="mean",
            phi=lambda x: x,
        )

        downloaded_model, exists = download_model(
            "Rainbow", "BreakoutNoFrameskip-v4", model_type=self.pretrained_type
        )
        agent.load(downloaded_model)
        if os.environ.get("PFRL_ASSERT_DOWNLOADED_MODEL_IS_CACHED"):
            assert exists

    def test_cpu(self):
        self._test_load_rainbow(gpu=None)

    @pytest.mark.gpu
    def test_gpu(self):
        self._test_load_rainbow(gpu=0)


@pytest.mark.parametrize("pretrained_type", ["final", "best"])
class TestLoadA3C:
    @pytest.fixture(autouse=True)
    def setup(self, pretrained_type):
        self.pretrained_type = pretrained_type

    def test_load_a3c(self):
        from pfrl.policies import SoftmaxCategoricalHead

        obs_size = 4
        n_actions = 4
        a3c_model = nn.Sequential(
            nn.Conv2d(obs_size, 16, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2592, 256),
            nn.ReLU(),
            pfrl.nn.Branched(
                nn.Sequential(
                    nn.Linear(256, n_actions),
                    SoftmaxCategoricalHead(),
                ),
                nn.Linear(256, 1),
            ),
        )
        from pfrl.optimizers import SharedRMSpropEpsInsideSqrt

        opt = SharedRMSpropEpsInsideSqrt(
            a3c_model.parameters(), lr=7e-4, eps=1e-1, alpha=0.99
        )
        agent = agents.A3C(
            a3c_model, opt, t_max=5, gamma=0.99, beta=1e-2, phi=lambda x: x
        )
        downloaded_model, exists = download_model(
            "A3C", "BreakoutNoFrameskip-v4", model_type=self.pretrained_type
        )
        agent.load(downloaded_model)
        if os.environ.get("PFRL_ASSERT_DOWNLOADED_MODEL_IS_CACHED"):
            assert exists


@pytest.mark.parametrize("pretrained_type", ["final", "best"])
class TestLoadDDPG:
    @pytest.fixture(autouse=True)
    def setup(self, pretrained_type):
        self.pretrained_type = pretrained_type

    def _test_load_ddpg(self, gpu):

        obs_size = 11
        action_size = 3
        from pfrl.nn import ConcatObsAndAction

        q_func = nn.Sequential(
            ConcatObsAndAction(),
            nn.Linear(obs_size + action_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )
        from pfrl.nn import BoundByTanh
        from pfrl.policies import DeterministicHead

        policy = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, action_size),
            BoundByTanh(low=[-1.0, -1.0, -1.0], high=[1.0, 1.0, 1.0]),
            DeterministicHead(),
        )

        opt_a = torch.optim.Adam(policy.parameters())
        opt_c = torch.optim.Adam(q_func.parameters())

        explorer = explorers.AdditiveGaussian(
            scale=0.1, low=[-1.0, -1.0, -1.0], high=[1.0, 1.0, 1.0]
        )

        agent = agents.DDPG(
            policy,
            q_func,
            opt_a,
            opt_c,
            replay_buffers.ReplayBuffer(100),
            gamma=0.99,
            explorer=explorer,
            replay_start_size=1000,
            target_update_method="soft",
            target_update_interval=1,
            update_interval=1,
            soft_update_tau=5e-3,
            n_times_update=1,
            gpu=gpu,
            minibatch_size=100,
            burnin_action_func=None,
        )

        downloaded_model, exists = download_model(
            "DDPG", "Hopper-v2", model_type=self.pretrained_type
        )
        agent.load(downloaded_model)
        if os.environ.get("PFRL_ASSERT_DOWNLOADED_MODEL_IS_CACHED"):
            assert exists

    def test_cpu(self):
        self._test_load_ddpg(gpu=None)

    @pytest.mark.gpu
    def test_gpu(self):
        self._test_load_ddpg(gpu=0)


@pytest.mark.parametrize("pretrained_type", ["final", "best"])
class TestLoadTRPO:
    @pytest.fixture(autouse=True)
    def setup(self, pretrained_type):
        self.pretrained_type = pretrained_type

    def _test_load_trpo(self, gpu):
        obs_size = 11
        action_size = 3

        policy = torch.nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_size),
            pfrl.policies.GaussianHeadWithStateIndependentCovariance(
                action_size=action_size,
                var_type="diagonal",
                var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
                var_param_init=0,  # log std = 0 => std = 1
            ),
        )

        vf = torch.nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        vf_opt = torch.optim.Adam(vf.parameters())

        agent = agents.TRPO(
            policy=policy,
            vf=vf,
            vf_optimizer=vf_opt,
            gpu=gpu,
            update_interval=5000,
            max_kl=0.01,
            conjugate_gradient_max_iter=20,
            conjugate_gradient_damping=1e-1,
            gamma=0.995,
            lambd=0.97,
            vf_epochs=5,
            entropy_coef=0,
        )

        downloaded_model, exists = download_model(
            "TRPO", "Hopper-v2", model_type=self.pretrained_type
        )
        agent.load(downloaded_model)
        if os.environ.get("PFRL_ASSERT_DOWNLOADED_MODEL_IS_CACHED"):
            assert exists

    def test_cpu(self):
        self._test_load_trpo(gpu=None)

    @pytest.mark.gpu
    def test_gpu(self):
        self._test_load_trpo(gpu=0)


class TestLoadPPO:
    def _test_load_ppo(self, gpu):
        obs_size = 11
        action_size = 3
        from pfrl.policies import GaussianHeadWithStateIndependentCovariance

        policy = torch.nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_size),
            GaussianHeadWithStateIndependentCovariance(
                action_size=action_size,
                var_type="diagonal",
                var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
                var_param_init=0,  # log std = 0 => std = 1
            ),
        )

        vf = torch.nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        model = pnn.Branched(policy, vf)
        opt = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)

        agent = agents.PPO(
            model,
            opt,
            obs_normalizer=None,
            gpu=gpu,
            update_interval=2048,
            minibatch_size=64,
            epochs=10,
            clip_eps_vf=None,
            entropy_coef=0,
            standardize_advantages=True,
            gamma=0.995,
            lambd=0.97,
        )

        downloaded_model, exists = download_model(
            "PPO", "Hopper-v2", model_type="final"
        )
        agent.load(downloaded_model)
        if os.environ.get("PFRL_ASSERT_DOWNLOADED_MODEL_IS_CACHED"):
            assert exists

    def test_cpu(self):
        self._test_load_ppo(gpu=None)

    @pytest.mark.gpu
    def test_gpu(self):
        self._test_load_ppo(gpu=0)


@pytest.mark.parametrize("pretrained_type", ["final", "best"])
class TestLoadTD3:
    @pytest.fixture(autouse=True)
    def setup(self, pretrained_type):
        self.pretrained_type = pretrained_type

    def _test_load_td3(self, gpu):

        obs_size = 11
        action_size = 3

        def make_q_func_with_optimizer():
            q_func = nn.Sequential(
                pnn.ConcatObsAndAction(),
                nn.Linear(obs_size + action_size, 400),
                nn.ReLU(),
                nn.Linear(400, 300),
                nn.ReLU(),
                nn.Linear(300, 1),
            )
            q_func_optimizer = torch.optim.Adam(q_func.parameters())
            return q_func, q_func_optimizer

        q_func1, q_func1_optimizer = make_q_func_with_optimizer()
        q_func2, q_func2_optimizer = make_q_func_with_optimizer()

        policy = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, action_size),
            nn.Tanh(),
            pfrl.policies.DeterministicHead(),
        )
        policy_optimizer = torch.optim.Adam(policy.parameters())

        rbuf = replay_buffers.ReplayBuffer(100)
        explorer = explorers.AdditiveGaussian(
            scale=0.1, low=[-1.0, -1.0, -1.0], high=[1.0, 1.0, 1.0]
        )

        agent = agents.TD3(
            policy,
            q_func1,
            q_func2,
            policy_optimizer,
            q_func1_optimizer,
            q_func2_optimizer,
            rbuf,
            gamma=0.99,
            soft_update_tau=5e-3,
            explorer=explorer,
            replay_start_size=1000,
            gpu=gpu,
            minibatch_size=100,
            burnin_action_func=None,
        )

        downloaded_model, exists = download_model(
            "TD3", "Hopper-v2", model_type=self.pretrained_type
        )
        agent.load(downloaded_model)
        if os.environ.get("PFRL_ASSERT_DOWNLOADED_MODEL_IS_CACHED"):
            assert exists

    def test_cpu(self):
        self._test_load_td3(gpu=None)

    @pytest.mark.gpu
    def test_gpu(self):
        self._test_load_td3(gpu=0)


@pytest.mark.parametrize("pretrained_type", ["final", "best"])
class TestLoadSAC:
    @pytest.fixture(autouse=True)
    def setup(self, pretrained_type):
        self.pretrained_type = pretrained_type

    def _test_load_sac(self, gpu):
        obs_size = 11
        action_size = 3

        def squashed_diagonal_gaussian_head(x):
            assert x.shape[-1] == action_size * 2
            mean, log_scale = torch.chunk(x, 2, dim=1)
            log_scale = torch.clamp(log_scale, -20.0, 2.0)
            var = torch.exp(log_scale * 2)
            from torch import distributions

            base_distribution = distributions.Independent(
                distributions.Normal(loc=mean, scale=torch.sqrt(var)), 1
            )
            # cache_size=1 is required for numerical stability
            return distributions.transformed_distribution.TransformedDistribution(
                base_distribution,
                [distributions.transforms.TanhTransform(cache_size=1)],
            )

        from pfrl.nn.lmbda import Lambda

        policy = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_size * 2),
            Lambda(squashed_diagonal_gaussian_head),
        )
        policy_optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

        def make_q_func_with_optimizer():
            q_func = nn.Sequential(
                pfrl.nn.ConcatObsAndAction(),
                nn.Linear(obs_size + action_size, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )
            torch.nn.init.xavier_uniform_(q_func[1].weight)
            torch.nn.init.xavier_uniform_(q_func[3].weight)
            torch.nn.init.xavier_uniform_(q_func[5].weight)
            q_func_optimizer = torch.optim.Adam(q_func.parameters(), lr=3e-4)
            return q_func, q_func_optimizer

        q_func1, q_func1_optimizer = make_q_func_with_optimizer()
        q_func2, q_func2_optimizer = make_q_func_with_optimizer()

        agent = agents.SoftActorCritic(
            policy,
            q_func1,
            q_func2,
            policy_optimizer,
            q_func1_optimizer,
            q_func2_optimizer,
            replay_buffers.ReplayBuffer(100),
            gamma=0.99,
            replay_start_size=1000,
            gpu=gpu,
            minibatch_size=256,
            burnin_action_func=None,
            entropy_target=-3,
            temperature_optimizer_lr=3e-4,
        )

        downloaded_model, exists = download_model(
            "SAC", "Hopper-v2", model_type=self.pretrained_type
        )
        agent.load(downloaded_model)
        if os.environ.get("PFRL_ASSERT_DOWNLOADED_MODEL_IS_CACHED"):
            assert exists

    def test_cpu(self):
        self._test_load_sac(gpu=None)

    @pytest.mark.gpu
    def test_gpu(self):
        self._test_load_sac(gpu=0)
