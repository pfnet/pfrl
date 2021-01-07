import unittest

import basetest_dqn_like as base
import numpy as np
import pytest
import torch
from basetest_training import _TestBatchTrainingMixin

import pfrl
from pfrl.agents import CategoricalDQN, categorical_dqn
from pfrl.agents.categorical_dqn import compute_value_loss, compute_weighted_value_loss

assertions = unittest.TestCase("__init__")


def _apply_categorical_projection_naive(y, y_probs, z):
    """Naively implemented categorical projection for checking results.

    See (7) in https://arxiv.org/abs/1802.08163.
    """
    batch_size, n_atoms = y.shape
    assert z.shape == (n_atoms,)
    assert y_probs.shape == (batch_size, n_atoms)
    v_min = z[0]
    v_max = z[-1]
    proj_probs = np.zeros((batch_size, n_atoms), dtype=np.float32)
    for b in range(batch_size):
        for i in range(n_atoms):
            yi = y[b, i]
            p = y_probs[b, i]
            if yi <= v_min:
                proj_probs[b, 0] += p
            elif yi > v_max:
                proj_probs[b, -1] += p
            else:
                for j in range(n_atoms - 1):
                    if z[j] < yi <= z[j + 1]:
                        delta_z = z[j + 1] - z[j]
                        proj_probs[b, j] += (z[j + 1] - yi) / delta_z * p
                        proj_probs[b, j + 1] += (yi - z[j]) / delta_z * p
                        break
                else:
                    assert False
    return proj_probs


@pytest.mark.parametrize(
    "batch_size",
    [1, 7],
)
@pytest.mark.parametrize(
    "n_atoms",
    [2, 5],
)
@pytest.mark.parametrize(
    "v_range",
    [(-3, -1), (-2, 0), (-2, 1), (0, 1), (1, 5)],
)
class TestApplyCategoricalProjectionToRandomCases:
    @pytest.fixture(autouse=True)
    def setup(self, batch_size, n_atoms, v_range):
        self.batch_size = batch_size
        self.n_atoms = n_atoms
        self.v_range = v_range

    def _test(self, device):
        v_min, v_max = self.v_range
        z = np.linspace(v_min, v_max, num=self.n_atoms, dtype=np.float32)
        y = np.random.normal(size=(self.batch_size, self.n_atoms)).astype(np.float32)
        y_probs = np.asarray(
            np.random.dirichlet(
                alpha=np.ones(self.n_atoms), size=self.batch_size
            ).astype(np.float32)
        )

        # Naive implementation as ground truths
        proj_gt = _apply_categorical_projection_naive(y, y_probs, z)
        # Projected probabilities should sum to one
        np.testing.assert_allclose(
            proj_gt.sum(axis=1), np.ones(self.batch_size, dtype=np.float32), atol=1e-5
        )

        # Batch implementation to test
        proj = (
            categorical_dqn._apply_categorical_projection(
                torch.as_tensor(y, device=device),
                torch.as_tensor(y_probs, device=device),
                torch.as_tensor(z, device=device),
            )
            .detach()
            .cpu()
            .numpy()
        )
        # Projected probabilities should sum to one
        np.testing.assert_allclose(
            proj.sum(axis=1), np.ones(self.batch_size, dtype=np.float32), atol=1e-5
        )

        # Both should be equal
        np.testing.assert_allclose(proj, proj_gt, atol=1e-5)

    def test_cpu(self):
        self._test(device=torch.device("cpu"))

    @pytest.mark.gpu
    def test_gpu(self):
        self._test(device=torch.device("cuda"))


class TestApplyCategoricalProjectionToManualCases(unittest.TestCase):
    def _test(self, device):
        v_min, v_max = (-1, 1)
        n_atoms = 3
        z = np.linspace(v_min, v_max, num=n_atoms, dtype=np.float32)
        y = np.asarray(
            [
                [-1, 0, 1],
                [1, -1, 0],
                [1, 1, 1],
                [-1, -1, -1],
                [0, 0, 0],
                [-0.5, 0, 1],
                [-0.5, 0, 0.5],
            ],
            dtype=np.float32,
        )
        y_probs = np.asarray(
            [
                [0.5, 0.2, 0.3],
                [0.5, 0.2, 0.3],
                [0.5, 0.2, 0.3],
                [0.5, 0.2, 0.3],
                [0.5, 0.2, 0.3],
                [0.5, 0.2, 0.3],
                [0.5, 0.2, 0.3],
            ],
            dtype=np.float32,
        )

        proj_gt = np.asarray(
            [
                [0.5, 0.2, 0.3],
                [0.2, 0.3, 0.5],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.25, 0.45, 0.3],
                [0.25, 0.6, 0.15],
            ],
            dtype=np.float32,
        )

        proj = (
            categorical_dqn._apply_categorical_projection(
                torch.as_tensor(y, device=device),
                torch.as_tensor(y_probs, device=device),
                torch.as_tensor(z, device=device),
            )
            .detach()
            .cpu()
            .numpy()
        )
        np.testing.assert_allclose(proj, proj_gt, atol=1e-5)

    def _test_inexact_delta_z(self, device):
        v_min, v_max = (-1, 1)
        n_atoms = 4
        # delta_z=2/3=0.66666... is not exact
        z = np.linspace(v_min, v_max, num=n_atoms, dtype=np.float32)
        y = np.asarray(
            [
                [-1, -1, 1, 1],
                [-1, 0, 1, 1],
            ],
            dtype=np.float32,
        )
        y_probs = np.asarray(
            [
                [0.5, 0.1, 0.1, 0.3],
                [0.5, 0.2, 0.0, 0.3],
            ],
            dtype=np.float32,
        )
        proj_gt = np.asarray(
            [
                [0.6, 0.0, 0.0, 0.4],
                [0.5, 0.1, 0.1, 0.3],
            ],
            dtype=np.float32,
        )

        proj = (
            categorical_dqn._apply_categorical_projection(
                torch.as_tensor(y, device=device),
                torch.as_tensor(y_probs, device=device),
                torch.as_tensor(z, device=device),
            )
            .detach()
            .cpu()
            .numpy()
        )
        np.testing.assert_allclose(proj, proj_gt, atol=1e-5)

    def test_cpu(self):
        self._test(device=torch.device("cpu"))

    @pytest.mark.gpu
    def test_gpu(self):
        self._test(device=torch.device("cuda"))

    def test_inexact_delta_z_cpu(self):
        self._test_inexact_delta_z(device=torch.device("cpu"))

    @pytest.mark.gpu
    def test_inexact_delta_z_gpu(self):
        self._test_inexact_delta_z(device=torch.device("cuda"))


def make_distrib_ff_q_func(env):
    n_atoms = 51
    v_max = 10
    v_min = -10
    return pfrl.q_functions.DistributionalFCStateQFunctionWithDiscreteAction(  # NOQA
        env.observation_space.low.size,
        env.action_space.n,
        n_atoms=n_atoms,
        v_min=v_min,
        v_max=v_max,
        n_hidden_channels=20,
        n_hidden_layers=1,
    )


def make_distrib_recurrent_q_func(env):
    n_atoms = 51
    v_max = 10
    v_min = -10
    return pfrl.nn.RecurrentSequential(
        torch.nn.RNN(input_size=env.observation_space.low.size, hidden_size=20),
        pfrl.q_functions.DistributionalFCStateQFunctionWithDiscreteAction(  # NOQA
            20,
            env.action_space.n,
            n_atoms=n_atoms,
            v_min=v_min,
            v_max=v_max,
            n_hidden_channels=None,
            n_hidden_layers=0,
        ),
    )


class TestCategoricalDQNOnDiscreteABC(
    _TestBatchTrainingMixin, base._TestDQNOnDiscreteABC
):
    def make_q_func(self, env):
        return make_distrib_ff_q_func(env)

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return CategoricalDQN(
            q_func,
            opt,
            rbuf,
            gpu=gpu,
            gamma=0.9,
            explorer=explorer,
            replay_start_size=100,
            target_update_interval=100,
        )


# Continuous action spaces are not supported
class TestCategoricalDQNOnDiscretePOABC(
    _TestBatchTrainingMixin, base._TestDQNOnDiscretePOABC
):
    def make_q_func(self, env):
        return make_distrib_recurrent_q_func(env)

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return CategoricalDQN(
            q_func,
            opt,
            rbuf,
            gpu=gpu,
            gamma=0.9,
            explorer=explorer,
            replay_start_size=100,
            target_update_interval=100,
            recurrent=True,
        )


def categorical_loss(y, t):
    return -t * np.log(np.clip(y, 1e-10, 1.0))


@pytest.mark.parametrize("batch_accumulator", ["mean", "sum"])
class TestComputeValueLoss:
    @pytest.fixture(autouse=True)
    def setUp(self, batch_accumulator):
        self.batch_accumulator = batch_accumulator
        # y and t are (batchsize, n_atoms)
        self.y = np.asarray([[0.1, 0.2, 0.3, 0.4], [0.05, 0.1, 0.2, 0.65]], dtype="f")
        self.t = np.asarray([[0.2, 0.2, 0.2, 0.4], [0.1, 0.3, 0.3, 0.3]], dtype="f")
        self.eltwise_losses = np.asarray(
            [categorical_loss(a, b) for a, b in zip(self.y, self.t)]
        )

    def test_not_weighted(self):
        loss = (
            compute_value_loss(
                torch.as_tensor(self.eltwise_losses),
                batch_accumulator=self.batch_accumulator,
            )
            .detach()
            .cpu()
            .numpy()
        )
        if self.batch_accumulator == "mean":
            eltwise_loss = self.eltwise_losses.sum(axis=1).mean()
        else:
            eltwise_loss = self.eltwise_losses.sum()
        assertions.assertAlmostEqual(loss, eltwise_loss, places=5)

    def test_uniformly_weighted(self):

        # Uniform weights of size batch size
        w1 = np.ones(self.y.shape[0], dtype="f")

        loss_w1 = (
            compute_weighted_value_loss(
                torch.as_tensor(self.eltwise_losses),
                self.y.shape[0],
                torch.as_tensor(w1),
                batch_accumulator=self.batch_accumulator,
            )
            .detach()
            .cpu()
            .numpy()
        )
        if self.batch_accumulator == "mean":
            eltwise_loss = self.eltwise_losses.sum(axis=1).mean()
        else:
            eltwise_loss = self.eltwise_losses.sum()
        assertions.assertAlmostEqual(loss_w1, eltwise_loss, places=5)

    def test_randomly_weighted(self):

        # Random weights
        wu = np.random.uniform(low=0, high=2, size=self.y.shape[0]).astype("f")

        loss_wu = (
            compute_weighted_value_loss(
                torch.as_tensor(self.eltwise_losses),
                self.y.shape[0],
                torch.as_tensor(wu),
                batch_accumulator=self.batch_accumulator,
            )
            .detach()
            .cpu()
            .numpy()
        )
        if self.batch_accumulator == "mean":
            eltwise_loss = (self.eltwise_losses.sum(axis=1) * wu).mean()
        else:
            eltwise_loss = (self.eltwise_losses * wu[:, None]).sum()
        assertions.assertAlmostEqual(loss_wu, eltwise_loss, places=5)
