import unittest

import basetest_dqn_like as base
import pytest
import torch
from basetest_training import _TestActorLearnerTrainingMixin, _TestBatchTrainingMixin

import pfrl
from pfrl.agents.dqn import DQN, compute_value_loss, compute_weighted_value_loss

assertions = unittest.TestCase("__init__")


class TestDQNOnDiscreteABC(
    _TestActorLearnerTrainingMixin, _TestBatchTrainingMixin, base._TestDQNOnDiscreteABC
):
    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return DQN(
            q_func,
            opt,
            rbuf,
            gpu=gpu,
            gamma=0.9,
            explorer=explorer,
            replay_start_size=100,
            target_update_interval=100,
        )

    def test_replay_capacity_checked(self):
        env, _ = self.make_env_and_successful_return(test=False)
        q_func = self.make_q_func(env)
        opt = self.make_optimizer(env, q_func)
        explorer = self.make_explorer(env)
        rbuf = pfrl.replay_buffers.ReplayBuffer(capacity=90)
        with pytest.raises(ValueError):
            self.make_dqn_agent(
                env=env, q_func=q_func, opt=opt, explorer=explorer, rbuf=rbuf, gpu=None
            )


class TestDQNOnDiscreteABCBoltzmann(
    _TestActorLearnerTrainingMixin, _TestBatchTrainingMixin, base._TestDQNOnDiscreteABC
):
    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        explorer = pfrl.explorers.Boltzmann()
        return DQN(
            q_func,
            opt,
            rbuf,
            gpu=gpu,
            gamma=0.9,
            explorer=explorer,
            replay_start_size=100,
            target_update_interval=100,
        )


class TestDQNOnContinuousABC(
    _TestActorLearnerTrainingMixin,
    _TestBatchTrainingMixin,
    base._TestDQNOnContinuousABC,
):
    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return DQN(
            q_func,
            opt,
            rbuf,
            gpu=gpu,
            gamma=0.9,
            explorer=explorer,
            replay_start_size=100,
            target_update_interval=100,
        )


class TestDQNOnDiscretePOABC(
    _TestActorLearnerTrainingMixin,
    _TestBatchTrainingMixin,
    base._TestDQNOnDiscretePOABC,
):
    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return DQN(
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


class TestNStepDQNOnDiscreteABC(base._TestNStepDQNOnDiscreteABC):
    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return DQN(
            q_func,
            opt,
            rbuf,
            gpu=gpu,
            gamma=0.9,
            explorer=explorer,
            replay_start_size=100,
            target_update_interval=100,
        )


class TestNStepDQNOnDiscreteABCBoltzmann(base._TestNStepDQNOnDiscreteABC):
    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        explorer = pfrl.explorers.Boltzmann()
        return DQN(
            q_func,
            opt,
            rbuf,
            gpu=gpu,
            gamma=0.9,
            explorer=explorer,
            replay_start_size=100,
            target_update_interval=100,
        )


class TestNStepDQNOnContinuousABC(base._TestNStepDQNOnContinuousABC):
    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return DQN(
            q_func,
            opt,
            rbuf,
            gpu=gpu,
            gamma=0.9,
            explorer=explorer,
            replay_start_size=100,
            target_update_interval=100,
        )


def _huber_loss_1(a):
    if abs(a) < 1:
        return 0.5 * a ** 2
    else:
        return abs(a) - 0.5


@pytest.mark.parametrize("batch_accumulator", ["mean", "sum"])
@pytest.mark.parametrize("clip_delta", [True, False])
class TestComputeValueLoss:
    @pytest.fixture(autouse=True)
    def setUp(self, clip_delta, batch_accumulator):
        self.clip_delta = clip_delta
        self.batch_accumulator = batch_accumulator

        self.y = torch.FloatTensor([1.0, 2.0, 3.0, 4.0])
        self.t = torch.FloatTensor([2.1, 2.2, 2.3, 2.4])
        if self.clip_delta:
            self.gt_losses = torch.FloatTensor(
                [_huber_loss_1(a) for a in self.y - self.t]
            )
        else:
            self.gt_losses = torch.FloatTensor([0.5 * a ** 2 for a in self.y - self.t])

    def test_not_weighted(self):
        loss = compute_value_loss(
            self.y,
            self.t,
            clip_delta=self.clip_delta,
            batch_accumulator=self.batch_accumulator,
        )
        if self.batch_accumulator == "mean":
            gt_loss = self.gt_losses.mean()
        else:
            gt_loss = self.gt_losses.sum()
        assertions.assertAlmostEqual(loss.numpy(), gt_loss.numpy(), places=5)

    def test_uniformly_weighted(self):
        # Uniform weights
        w1 = torch.ones(self.y.size())

        loss_w1 = compute_weighted_value_loss(
            self.y,
            self.t,
            clip_delta=self.clip_delta,
            batch_accumulator=self.batch_accumulator,
            weights=w1,
        )
        if self.batch_accumulator == "mean":
            gt_loss = self.gt_losses.mean()
        else:
            gt_loss = self.gt_losses.sum()
        assertions.assertAlmostEqual(loss_w1.numpy(), gt_loss.numpy(), places=5)

    def test_randomly_weighted(self):
        # Random weights
        wu = torch.empty(self.y.size())
        torch.nn.init.uniform_(wu, a=0, b=2)

        loss_wu = compute_weighted_value_loss(
            self.y,
            self.t,
            clip_delta=self.clip_delta,
            batch_accumulator=self.batch_accumulator,
            weights=wu,
        )
        if self.batch_accumulator == "mean":
            gt_loss = (self.gt_losses * wu).mean()
        else:
            gt_loss = (self.gt_losses * wu).sum()
        assertions.assertAlmostEqual(loss_wu.numpy(), gt_loss.numpy(), places=5)
