import basetest_dqn_like
from basetest_training import _TestBatchTrainingMixin

from pfrl.agents.double_pal import DoublePAL


class TestDoublePALOnDiscreteABC(
    _TestBatchTrainingMixin, basetest_dqn_like._TestDQNOnDiscreteABC
):
    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return DoublePAL(
            q_func,
            opt,
            rbuf,
            gpu=gpu,
            gamma=0.9,
            explorer=explorer,
            replay_start_size=100,
            target_update_interval=100,
        )


class TestDoublePALOnContinuousABC(
    _TestBatchTrainingMixin, basetest_dqn_like._TestDQNOnContinuousABC
):
    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return DoublePAL(
            q_func,
            opt,
            rbuf,
            gpu=gpu,
            gamma=0.9,
            explorer=explorer,
            replay_start_size=100,
            target_update_interval=100,
        )


class TestDoublePALOnDiscretePOABC(
    _TestBatchTrainingMixin, basetest_dqn_like._TestDQNOnDiscretePOABC
):
    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return DoublePAL(
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
