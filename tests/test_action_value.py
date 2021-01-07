import unittest

import numpy as np
import pytest
import torch

from pfrl import action_value
from pfrl.testing import torch_assert_allclose

assertions = unittest.TestCase("__init__")


class TestDiscreteActionValue:
    @pytest.fixture(autouse=True)
    def setUp(self):
        self.batch_size = 30
        self.action_size = 3
        self.q_values = np.random.normal(
            size=(self.batch_size, self.action_size)
        ).astype(np.float32)
        self.qout = action_value.DiscreteActionValue(torch.from_numpy(self.q_values))

    def test_max(self):
        assertions.assertIsInstance(self.qout.max, torch.Tensor)
        np.testing.assert_almost_equal(self.qout.max.numpy(), self.q_values.max(axis=1))

    def test_greedy_actions(self):
        assertions.assertIsInstance(self.qout.greedy_actions, torch.Tensor)
        np.testing.assert_equal(
            self.qout.greedy_actions.numpy(), self.q_values.argmax(axis=1)
        )

    def test_evaluate_actions(self):
        sample_actions = np.random.randint(self.action_size, size=self.batch_size)
        sample_actions = torch.from_numpy(sample_actions)
        ret = self.qout.evaluate_actions(sample_actions)
        assertions.assertIsInstance(ret, torch.Tensor)
        for b in range(self.batch_size):
            assertions.assertAlmostEqual(
                ret.numpy()[b], self.q_values[b, sample_actions[b]]
            )

    def test_compute_advantage(self):
        sample_actions = np.random.randint(self.action_size, size=self.batch_size)
        greedy_actions = self.q_values.argmax(axis=1)
        sample_actions = torch.from_numpy(sample_actions)
        ret = self.qout.compute_advantage(sample_actions)
        assertions.assertIsInstance(ret, torch.Tensor)
        for b in range(self.batch_size):
            if sample_actions[b] == greedy_actions[b]:
                assertions.assertAlmostEqual(ret.numpy()[b], 0)
            else:
                # An advantage to the optimal policy must be always negative
                assertions.assertLess(ret.numpy()[b], 0)
                q = self.q_values[b, sample_actions[b]]
                v = self.q_values[b, greedy_actions[b]]
                adv = q - v
                assertions.assertAlmostEqual(ret.numpy()[b], adv)

    def test_params(self):
        assertions.assertEqual(len(self.qout.params), 1)
        assertions.assertEqual(id(self.qout.params[0]), id(self.qout.q_values))

    def test_getitem(self):
        sliced = self.qout[:10]
        np.testing.assert_equal(sliced.q_values.numpy(), self.q_values[:10])
        assertions.assertEqual(sliced.n_actions, self.action_size)
        assertions.assertIs(sliced.q_values_formatter, self.qout.q_values_formatter)


class TestDistributionalDiscreteActionValue(unittest.TestCase):
    def setUp(self):
        self.batch_size = 30
        self.action_size = 3
        self.n_atoms = 51
        self.atom_probs = np.random.dirichlet(
            alpha=np.ones(self.n_atoms), size=(self.batch_size, self.action_size)
        ).astype(np.float32)
        self.z_values = np.linspace(-10, 10, num=self.n_atoms, dtype=np.float32)
        self.qout = action_value.DistributionalDiscreteActionValue(
            torch.as_tensor(self.atom_probs), torch.as_tensor(self.z_values)
        )
        self.q_values = (self.atom_probs * self.z_values).sum(axis=2)

    def test_max(self):
        self.assertIsInstance(self.qout.max, torch.Tensor)
        np.testing.assert_almost_equal(
            self.qout.max.detach().cpu().numpy(), self.q_values.max(axis=1), decimal=5
        )

    def test_max_as_distribution(self):
        self.assertIsInstance(self.qout.max_as_distribution, torch.Tensor)
        for b in range(self.batch_size):
            np.testing.assert_almost_equal(
                self.qout.max_as_distribution.detach().cpu().numpy()[b],
                self.atom_probs[b, self.qout.greedy_actions.detach().cpu().numpy()[b]],
                decimal=5,
            )

    def test_greedy_actions(self):
        self.assertIsInstance(self.qout.greedy_actions, torch.Tensor)
        np.testing.assert_almost_equal(
            self.qout.greedy_actions.detach().cpu().numpy(),
            self.q_values.argmax(axis=1),
            decimal=5,
        )

    def test_evaluate_actions(self):
        sample_actions = torch.as_tensor(
            np.random.randint(self.action_size, size=self.batch_size)
        )
        ret = self.qout.evaluate_actions(sample_actions)
        self.assertIsInstance(ret, torch.Tensor)
        for b in range(self.batch_size):
            self.assertAlmostEqual(
                ret.detach().cpu().numpy()[b],
                self.q_values[b, sample_actions[b]],
                places=5,
            )

    def test_evaluate_actions_as_distribution(self):
        sample_actions = torch.as_tensor(
            np.random.randint(self.action_size, size=self.batch_size)
        )
        ret = self.qout.evaluate_actions_as_distribution(sample_actions)
        self.assertIsInstance(ret, torch.Tensor)
        for b in range(self.batch_size):
            np.testing.assert_almost_equal(
                ret.detach().cpu().numpy()[b],
                self.atom_probs[b, sample_actions[b]],
                decimal=5,
            )

    def test_compute_advantage(self):
        sample_actions = torch.as_tensor(
            np.random.randint(self.action_size, size=self.batch_size)
        )
        greedy_actions = self.q_values.argmax(axis=1)
        ret = self.qout.compute_advantage(sample_actions)
        self.assertIsInstance(ret, torch.Tensor)
        for b in range(self.batch_size):
            if sample_actions[b] == greedy_actions[b]:
                self.assertAlmostEqual(ret.detach().cpu().numpy()[b], 0, places=5)
            else:
                # An advantage to the optimal policy must be always negative
                self.assertLess(ret.detach().cpu().numpy()[b], 0)
                q = self.q_values[b, sample_actions[b]]
                v = self.q_values[b, greedy_actions[b]]
                adv = q - v
                self.assertAlmostEqual(ret.detach().cpu().numpy()[b], adv, places=5)

    def test_params(self):
        self.assertEqual(len(self.qout.params), 1)
        self.assertIs(self.qout.params[0], self.qout.q_dist)

    def test_getitem(self):
        sliced = self.qout[:10]
        np.testing.assert_almost_equal(
            sliced.q_values.detach().cpu().numpy(), self.q_values[:10], decimal=5
        )
        np.testing.assert_almost_equal(sliced.z_values, self.z_values, decimal=5)
        np.testing.assert_almost_equal(
            sliced.q_dist.detach().cpu().numpy(), self.atom_probs[:10], decimal=5
        )
        self.assertEqual(sliced.n_actions, self.action_size)
        self.assertIs(sliced.q_values_formatter, self.qout.q_values_formatter)


class TestQuantileDiscreteActionValue(unittest.TestCase):
    def setUp(self):
        self.batch_size = 30
        self.action_size = 3
        self.n_taus = 5
        self.quantiles = torch.randn(
            self.batch_size,
            self.n_taus,
            self.action_size,
            dtype=torch.float,
        )
        self.av = action_value.QuantileDiscreteActionValue(self.quantiles)
        self.q_values = self.quantiles.mean(axis=1)

    def test_q_values(self):
        self.assertIsInstance(self.av.q_values, torch.Tensor)
        torch_assert_allclose(self.av.q_values, self.q_values)

    def test_evaluate_actions_as_quantiles(self):
        sample_actions = torch.randint(self.action_size, size=(self.batch_size,))
        z = self.av.evaluate_actions_as_quantiles(sample_actions)
        self.assertIsInstance(z, torch.Tensor)
        for b in range(self.batch_size):
            torch_assert_allclose(z[b], self.quantiles[b, :, sample_actions[b]])

    def test_params(self):
        self.assertEqual(len(self.av.params), 1)
        self.assertIs(self.av.params[0], self.av.quantiles)

    def test_getitem(self):
        sliced = self.av[:10]
        torch_assert_allclose(sliced.q_values, self.q_values[:10])
        torch_assert_allclose(sliced.quantiles, self.quantiles[:10])
        self.assertEqual(sliced.n_actions, self.action_size)
        self.assertIs(sliced.q_values_formatter, self.av.q_values_formatter)


class TestQuadraticActionValue(unittest.TestCase):
    def test_max_unbounded(self):
        n_batch = 7
        ndim_action = 3
        mu = np.random.randn(n_batch, ndim_action).astype(np.float32)
        mat = np.broadcast_to(
            np.eye(ndim_action, dtype=np.float32)[None],
            (n_batch, ndim_action, ndim_action),
        )
        v = np.random.randn(n_batch).astype(np.float32)
        q_out = action_value.QuadraticActionValue(
            torch.tensor(mu), torch.tensor(mat), torch.tensor(v)
        )

        v_out = q_out.max
        self.assertIsInstance(v_out, torch.Tensor)
        v_out = v_out.detach().numpy()

        np.testing.assert_almost_equal(v_out, v)

    def test_max_bounded(self):
        n_batch = 20
        ndim_action = 3
        mu = np.random.randn(n_batch, ndim_action).astype(np.float32)
        mat = np.broadcast_to(
            np.eye(ndim_action, dtype=np.float32)[None],
            (n_batch, ndim_action, ndim_action),
        )
        v = np.random.randn(n_batch).astype(np.float32)
        min_action, max_action = -1.3, 1.3
        q_out = action_value.QuadraticActionValue(
            torch.tensor(mu), torch.tensor(mat), torch.tensor(v), min_action, max_action
        )

        v_out = q_out.max
        self.assertIsInstance(v_out, torch.Tensor)
        v_out = v_out.detach().numpy()

        # If mu[i] is an valid action, v_out[i] should be v[i]
        mu_is_allowed = np.all((min_action < mu) * (mu < max_action), axis=1)
        np.testing.assert_almost_equal(v_out[mu_is_allowed], v[mu_is_allowed])

        # Otherwise, v_out[i] should be less than v[i]
        mu_is_not_allowed = ~np.all(
            (min_action - 1e-2 < mu) * (mu < max_action + 1e-2), axis=1
        )
        np.testing.assert_array_less(v_out[mu_is_not_allowed], v[mu_is_not_allowed])

    def test_getitem(self):
        n_batch = 7
        ndim_action = 3
        mu = np.random.randn(n_batch, ndim_action).astype(np.float32)
        mat = np.broadcast_to(
            np.eye(ndim_action, dtype=np.float32)[None],
            (n_batch, ndim_action, ndim_action),
        )
        v = np.random.randn(n_batch).astype(np.float32)
        min_action, max_action = -1, 1
        qout = action_value.QuadraticActionValue(
            torch.tensor(mu),
            torch.tensor(mat),
            torch.tensor(v),
            min_action,
            max_action,
        )
        sliced = qout[:3]
        torch_assert_allclose(sliced.mu, mu[:3])
        torch_assert_allclose(sliced.mat, mat[:3])
        torch_assert_allclose(sliced.v, v[:3])
        torch_assert_allclose(sliced.min_action[0], min_action)
        torch_assert_allclose(sliced.max_action[0], max_action)


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("action_size", [1, 2])
@pytest.mark.parametrize("has_maximizer", [True, False])
class TestSingleActionValue:
    @pytest.fixture(autouse=True)
    def setUp(self, batch_size, action_size, has_maximizer):
        self.batch_size = batch_size
        self.action_size = action_size
        self.has_maximizer = has_maximizer

        def evaluator(actions):
            # negative square norm of actions
            return -torch.sum(actions ** 2, dim=1)

        self.evaluator = evaluator

        if self.has_maximizer:

            def maximizer():
                return torch.from_numpy(
                    np.zeros((self.batch_size, self.action_size), dtype=np.float32)
                )

        else:
            maximizer = None
        self.maximizer = maximizer
        self.av = action_value.SingleActionValue(
            evaluator=evaluator, maximizer=maximizer
        )

    def test_max(self):
        if not self.has_maximizer:
            return
        assertions.assertIsInstance(self.av.max, torch.Tensor)
        np.testing.assert_almost_equal(
            self.av.max.numpy(), self.evaluator(self.maximizer()).numpy()
        )

    def test_greedy_actions(self):
        if not self.has_maximizer:
            return
        assertions.assertIsInstance(self.av.greedy_actions, torch.Tensor)
        np.testing.assert_equal(
            self.av.greedy_actions.numpy(), self.maximizer().numpy()
        )

    def test_evaluate_actions(self):
        sample_actions = np.random.randn(self.batch_size, self.action_size).astype(
            np.float32
        )
        sample_actions = torch.from_numpy(sample_actions)
        ret = self.av.evaluate_actions(sample_actions)
        assertions.assertIsInstance(ret, torch.Tensor)
        np.testing.assert_equal(ret.numpy(), self.evaluator(sample_actions).numpy())

    def test_compute_advantage(self):
        if not self.has_maximizer:
            return
        sample_actions = np.random.randn(self.batch_size, self.action_size).astype(
            np.float32
        )
        sample_actions = torch.from_numpy(sample_actions)
        ret = self.av.compute_advantage(sample_actions)
        assertions.assertIsInstance(ret, torch.Tensor)
        np.testing.assert_equal(
            ret.numpy(),
            (
                self.evaluator(sample_actions).numpy()
                - self.evaluator(self.maximizer()).numpy()
            ),
        )

    def test_params(self):
        # no params
        assertions.assertEqual(len(self.av.params), 0)

    def test_getitem(self):
        with assertions.assertRaises(NotImplementedError):
            self.av[:1]
