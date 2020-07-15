import functools
import os

import numpy as np
import pytest

import pfrl
from pfrl import agents
from pfrl.utils import download_model

pytestmark = pytest.mark.skip()


@pytest.mark.parametrize("pretrained_type", ["final", "best"])
class TestLoadA3C:
    @pytest.fixture(autouse=True)
    def setup(self, pretrained_type):
        self.pretrained_type = pretrained_type

    def _test_load_a3c(self, gpu):
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
            model.parameters(), lr=7e-4, eps=1e-1, alpha=0.99
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

    def test_cpu(self):
        self._test_load_a3c(gpu=None)

    @pytest.mark.gpu
    def test_gpu(self):
        self._test_load_a3c(gpu=0)
