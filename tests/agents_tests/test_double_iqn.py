from torch import nn
import pytest

import basetest_dqn_like as base

from basetest_training import _TestBatchTrainingMixin
import pfrl
from pfrl.agents import double_iqn, iqn


@pytest.mark.parametrize("quantile_thresholds_N", [1, 5])
@pytest.mark.parametrize("quantile_thresholds_N_prime", [1, 7])
class TestDoubleIQNOnDiscreteABC(
    _TestBatchTrainingMixin, base._TestDQNOnDiscreteABC,
):
    @pytest.fixture(autouse=True)
    def set_iqn_params(self, quantile_thresholds_N, quantile_thresholds_N_prime):
        self.quantile_thresholds_N = quantile_thresholds_N
        self.quantile_thresholds_N_prime = quantile_thresholds_N_prime

    def make_q_func(self, env):
        obs_size = env.observation_space.low.size
        hidden_size = 64
        return iqn.ImplicitQuantileQFunction(
            psi=nn.Sequential(nn.Linear(obs_size, hidden_size), nn.ReLU(),),
            phi=nn.Sequential(
                pfrl.agents.iqn.CosineBasisLinear(32, hidden_size), nn.ReLU(),
            ),
            f=nn.Linear(hidden_size, env.action_space.n),
        )

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return double_iqn.DoubleIQN(
            q_func,
            opt,
            rbuf,
            gpu=gpu,
            gamma=0.9,
            explorer=explorer,
            replay_start_size=100,
            target_update_interval=100,
            quantile_thresholds_N=self.quantile_thresholds_N,
            quantile_thresholds_N_prime=self.quantile_thresholds_N_prime,
            act_deterministically=True,
        )


class TestDoubleIQNOnDiscretePOABC(
    _TestBatchTrainingMixin, base._TestDQNOnDiscretePOABC,
):
    def make_q_func(self, env):
        obs_size = env.observation_space.low.size
        hidden_size = 64
        return iqn.RecurrentImplicitQuantileQFunction(
            psi=pfrl.nn.RecurrentSequential(
                nn.Linear(obs_size, hidden_size),
                nn.ReLU(),
                nn.RNN(num_layers=1, input_size=hidden_size, hidden_size=hidden_size,),
            ),
            phi=nn.Sequential(
                pfrl.agents.iqn.CosineBasisLinear(32, hidden_size), nn.ReLU(),
            ),
            f=nn.Linear(hidden_size, env.action_space.n),
        )

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return double_iqn.DoubleIQN(
            q_func,
            opt,
            rbuf,
            gpu=gpu,
            gamma=0.9,
            explorer=explorer,
            replay_start_size=100,
            target_update_interval=100,
            quantile_thresholds_N=32,
            quantile_thresholds_N_prime=32,
            recurrent=True,
            act_deterministically=True,
        )
