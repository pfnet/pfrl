import basetest_dqn_like as base
from basetest_training import _TestBatchTrainingMixin

from pfrl.agents.pal import PAL


class TestPALOnDiscreteABC(_TestBatchTrainingMixin, base._TestDQNOnDiscreteABC):
    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return PAL(
            q_func,
            opt,
            rbuf,
            gpu=gpu,
            gamma=0.9,
            explorer=explorer,
            replay_start_size=100,
            target_update_interval=100,
        )


class TestPALOnContinuousABC(_TestBatchTrainingMixin, base._TestDQNOnContinuousABC):
    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return PAL(
            q_func,
            opt,
            rbuf,
            gpu=gpu,
            gamma=0.9,
            explorer=explorer,
            replay_start_size=100,
            target_update_interval=100,
        )


class TestPALOnDiscretePOABC(_TestBatchTrainingMixin, base._TestDQNOnDiscretePOABC):
    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        return PAL(
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
