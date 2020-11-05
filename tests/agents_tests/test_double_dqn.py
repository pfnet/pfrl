import basetest_dqn_like
from basetest_training import _TestBatchTrainingMixin

from pfrl.agents.double_dqn import DoubleDQN


class TestDoubleDQNOnDiscreteABC(
    _TestBatchTrainingMixin, basetest_dqn_like._TestDQNOnDiscreteABC
):
    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return DoubleDQN(
            q_func,
            opt,
            rbuf,
            gpu=gpu,
            gamma=0.9,
            explorer=explorer,
            replay_start_size=100,
            target_update_interval=100,
        )


class TestDoubleDQNOnContinuousABC(
    _TestBatchTrainingMixin, basetest_dqn_like._TestDQNOnContinuousABC
):
    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return DoubleDQN(
            q_func,
            opt,
            rbuf,
            gpu=gpu,
            gamma=0.9,
            explorer=explorer,
            replay_start_size=100,
            target_update_interval=100,
        )


class TestDoubleDQNOnDiscretePOABC(
    _TestBatchTrainingMixin, basetest_dqn_like._TestDQNOnDiscretePOABC
):
    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return DoubleDQN(
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
