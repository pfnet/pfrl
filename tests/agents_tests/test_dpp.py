import basetest_dqn_like as base
import pytest
from basetest_training import _TestBatchTrainingMixin

from pfrl.agents.dpp import DPP, DPPL, DPPGreedy


def parse_dpp_agent(dpp_type):
    return {"DPP": DPP, "DPPL": DPPL, "DPPGreedy": DPPGreedy}[dpp_type]


@pytest.mark.parametrize("dpp_type", ["DPP", "DPPL", "DPPGreedy"])
class TestDPPOnDiscreteABC(_TestBatchTrainingMixin, base._TestDQNOnDiscreteABC):
    @pytest.fixture(autouse=True)
    def setUp(self, dpp_type):
        self.dpp_type = dpp_type

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        agent_class = parse_dpp_agent(self.dpp_type)
        return agent_class(
            q_func,
            opt,
            rbuf,
            gpu=gpu,
            gamma=0.9,
            explorer=explorer,
            replay_start_size=100,
            target_update_interval=100,
        )


# DPP and DPPL don't support continuous action spaces
@pytest.mark.parametrize("dpp_type", ["DPPGreedy"])
class TestDPPOnContinuousABC(_TestBatchTrainingMixin, base._TestDQNOnContinuousABC):
    @pytest.fixture(autouse=True)
    def setUp(self, dpp_type):
        self.dpp_type = dpp_type

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        agent_class = parse_dpp_agent(self.dpp_type)
        return agent_class(
            q_func,
            opt,
            rbuf,
            gpu=gpu,
            gamma=0.9,
            explorer=explorer,
            replay_start_size=100,
            target_update_interval=100,
        )


@pytest.mark.parametrize("dpp_type", ["DPP", "DPPL", "DPPGreedy"])
class TestDPPOnDiscretePOABC(_TestBatchTrainingMixin, base._TestDQNOnDiscretePOABC):
    @pytest.fixture(autouse=True)
    def setUp(self, dpp_type):
        self.dpp_type = dpp_type

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        agent_class = parse_dpp_agent(self.dpp_type)
        return agent_class(
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
