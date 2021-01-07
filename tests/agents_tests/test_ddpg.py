import basetest_ddpg as base
import pytest
from basetest_training import _TestBatchTrainingMixin

from pfrl.agents.ddpg import DDPG


@pytest.mark.skip  # recurrent=True is not supported yet
# Batch training with recurrent models is currently not supported
class TestDDPGOnContinuousPOABC(base._TestDDPGOnContinuousPOABC):
    def make_ddpg_agent(
        self,
        env,
        policy,
        q_func,
        actor_opt,
        critic_opt,
        explorer,
        rbuf,
        gpu,
    ):
        return DDPG(
            policy,
            q_func,
            actor_opt,
            critic_opt,
            rbuf,
            gpu=gpu,
            gamma=0.9,
            explorer=explorer,
            replay_start_size=100,
            target_update_method="soft",
            target_update_interval=1,
            recurrent=True,
            update_interval=1,
        )


class TestDDPGOnContinuousABC(_TestBatchTrainingMixin, base._TestDDPGOnContinuousABC):
    def make_ddpg_agent(
        self,
        env,
        policy,
        q_func,
        actor_opt,
        critic_opt,
        explorer,
        rbuf,
        gpu,
    ):
        return DDPG(
            policy,
            q_func,
            actor_opt,
            critic_opt,
            rbuf,
            gpu=gpu,
            gamma=0.9,
            explorer=explorer,
            replay_start_size=100,
            target_update_method="soft",
            target_update_interval=1,
            recurrent=False,
        )
