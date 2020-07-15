import os

import numpy as np
import pytest
import torch
from torch import nn

import pfrl
import pfrl.nn as pnn
from pfrl import agents
from pfrl import explorers
from pfrl import replay_buffers
from pfrl.utils import download_model
from pfrl.initializers import init_chainer_default


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
        if os.environ.get("CHAINERRL_ASSERT_DOWNLOADED_MODEL_IS_CACHED"):
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
            phi=nn.Sequential(pfrl.agents.iqn.CosineBasisLinear(64, 3136), nn.ReLU(),),
            f=nn.Sequential(
                nn.Linear(3136, 512), nn.ReLU(), nn.Linear(512, n_actions),
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
        if os.environ.get("CHAINERRL_ASSERT_DOWNLOADED_MODEL_IS_CACHED"):
            assert exists

    def test_cpu(self):
        self._test_load_iqn(gpu=None)

    @pytest.mark.gpu
    def test_gpu(self):
        self._test_load_iqn(gpu=0)


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
                nn.Sequential(nn.Linear(256, n_actions), SoftmaxCategoricalHead(),),
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
