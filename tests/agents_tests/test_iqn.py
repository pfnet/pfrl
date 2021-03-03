import basetest_dqn_like as base
import numpy as np
import pytest
import torch

# IQN does not support the actor-learner interface for now
# from basetest_training import _TestActorLearnerTrainingMixin
from basetest_training import _TestBatchTrainingMixin
from torch import nn

import pfrl
from pfrl.agents import iqn
from pfrl.testing import torch_assert_allclose


@pytest.mark.parametrize("quantile_thresholds_N", [1, 5])
@pytest.mark.parametrize("quantile_thresholds_N_prime", [1, 7])
class TestIQNOnDiscreteABC(
    # _TestActorLearnerTrainingMixin,
    _TestBatchTrainingMixin,
    base._TestDQNOnDiscreteABC,
):
    @pytest.fixture(autouse=True)
    def set_iqn_params(self, quantile_thresholds_N, quantile_thresholds_N_prime):
        self.quantile_thresholds_N = quantile_thresholds_N
        self.quantile_thresholds_N_prime = quantile_thresholds_N_prime

    def make_q_func(self, env):
        obs_size = env.observation_space.low.size
        hidden_size = 64
        return iqn.ImplicitQuantileQFunction(
            psi=nn.Sequential(
                nn.Linear(obs_size, hidden_size),
                nn.ReLU(),
            ),
            phi=nn.Sequential(
                pfrl.agents.iqn.CosineBasisLinear(32, hidden_size),
                nn.ReLU(),
            ),
            f=nn.Linear(hidden_size, env.action_space.n),
        )

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return iqn.IQN(
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


class TestIQNOnDiscretePOABC(
    # _TestActorLearnerTrainingMixin,
    _TestBatchTrainingMixin,
    base._TestDQNOnDiscretePOABC,
):
    def make_q_func(self, env):
        obs_size = env.observation_space.low.size
        hidden_size = 64
        return iqn.RecurrentImplicitQuantileQFunction(
            psi=pfrl.nn.RecurrentSequential(
                nn.Linear(obs_size, hidden_size),
                nn.ReLU(),
                nn.RNN(
                    num_layers=1,
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                ),
            ),
            phi=nn.Sequential(
                pfrl.agents.iqn.CosineBasisLinear(32, hidden_size),
                nn.ReLU(),
            ),
            f=nn.Linear(hidden_size, env.action_space.n),
        )

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return iqn.IQN(
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


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("N", [1, 5])
@pytest.mark.parametrize("N_prime", [1, 7])
def test_compute_eltwise_huber_quantile_loss(batch_size, N, N_prime):
    # Overestimation is penalized proportionally to 1-tau
    # Underestimation is penalized proportionally to tau
    y = torch.randn(batch_size, N, dtype=torch.float, requires_grad=True)
    t = torch.randn(batch_size, N_prime, dtype=torch.float)
    tau = torch.rand(batch_size, N, dtype=torch.float)

    loss = iqn.compute_eltwise_huber_quantile_loss(y, t, tau)
    y_b, t_b = torch.broadcast_tensors(
        y.reshape(batch_size, N, 1),
        t.reshape(batch_size, 1, N_prime),
    )
    assert loss.shape == (batch_size, N, N_prime)
    huber_loss = nn.functional.smooth_l1_loss(y_b, t_b, reduction="none")
    assert huber_loss.shape == (batch_size, N, N_prime)

    for i in range(batch_size):
        for j in range(N):
            for k in range(N_prime):
                # loss is always positive
                scalar_loss = loss[i, j, k]
                scalar_grad = torch.autograd.grad(
                    [scalar_loss], [y], retain_graph=True
                )[0][i, j]
                assert float(scalar_loss) > 0
                if y[i, j] > t[i, k]:
                    # y over-estimates t
                    # loss equals huber loss scaled by (1-tau)
                    correct_scalar_loss = (1 - tau[i, j]) * huber_loss[i, j, k]
                else:
                    # y under-estimates t
                    # loss equals huber loss scaled by tau
                    correct_scalar_loss = tau[i, j] * huber_loss[i, j, k]
                correct_scalar_grad = torch.autograd.grad(
                    [correct_scalar_loss], [y], retain_graph=True
                )[0][i, j]
                torch_assert_allclose(
                    scalar_loss,
                    correct_scalar_loss,
                    atol=1e-5,
                )
                torch_assert_allclose(
                    scalar_grad,
                    correct_scalar_grad,
                    atol=1e-5,
                )


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("m", [1, 5])
@pytest.mark.parametrize("n_basis_functions", [1, 7])
def test_cosine_basis_functions(batch_size, m, n_basis_functions):
    x = torch.rand(batch_size, m, dtype=torch.float)
    y = iqn.cosine_basis_functions(x, n_basis_functions=n_basis_functions)
    assert y.shape == (batch_size, m, n_basis_functions)

    for i in range(batch_size):
        for j in range(m):
            for k in range(n_basis_functions):
                torch_assert_allclose(
                    y[i, j, k],
                    torch.cos(x[i, j] * (k + 1) * np.pi),
                    atol=1e-5,
                )
