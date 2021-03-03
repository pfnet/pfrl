from abc import ABCMeta, abstractmethod

import torch

from pfrl.agents.dqn import DQN
from pfrl.utils.recurrent import pack_and_forward


class AbstractDPP(DQN, metaclass=ABCMeta):
    """Dynamic Policy Programming.

    See: https://arxiv.org/abs/1004.2027.
    """

    @abstractmethod
    def _l_operator(self, qout):
        raise NotImplementedError()

    def _compute_target_values(self, exp_batch):

        batch_next_state = exp_batch["next_state"]

        if self.recurrent:
            target_next_qout, _ = pack_and_forward(
                self.target_model,
                batch_next_state,
                exp_batch["next_recurrent_state"],
            )
        else:
            target_next_qout = self.target_model(batch_next_state)
        next_q_expect = self._l_operator(target_next_qout)

        batch_rewards = exp_batch["reward"]
        batch_terminal = exp_batch["is_state_terminal"]

        return (
            batch_rewards + exp_batch["discount"] * (1 - batch_terminal) * next_q_expect
        )

    def _compute_y_and_t(self, exp_batch):

        batch_state = exp_batch["state"]
        batch_size = len(exp_batch["reward"])

        if self.recurrent:
            qout, _ = pack_and_forward(
                self.model,
                batch_state,
                exp_batch["recurrent_state"],
            )
        else:
            qout = self.model(batch_state)

        batch_actions = exp_batch["action"]
        # Q(s_t,a_t)
        batch_q = qout.evaluate_actions(batch_actions).reshape((batch_size, 1))

        with torch.no_grad():
            # Compute target values
            if self.recurrent:
                target_qout, _ = pack_and_forward(
                    self.target_model,
                    batch_state,
                    exp_batch["recurrent_state"],
                )
            else:
                target_qout = self.target_model(batch_state)

            # Q'(s_t,a_t)
            target_q = target_qout.evaluate_actions(batch_actions).reshape(
                (batch_size, 1)
            )

            # LQ'(s_t,a)
            target_q_expect = self._l_operator(target_qout).reshape((batch_size, 1))

            # r + g * LQ'(s_{t+1},a)
            batch_q_target = self._compute_target_values(exp_batch).reshape(
                (batch_size, 1)
            )

            # Q'(s_t,a_t) + r + g * LQ'(s_{t+1},a) - LQ'(s_t,a)
            t = target_q + batch_q_target - target_q_expect

        return batch_q, t


class DPP(AbstractDPP):
    """Dynamic Policy Programming with softmax operator.

    Args:
      eta (float): Positive constant.

    For other arguments, see DQN.
    """

    def __init__(self, *args, **kwargs):
        self.eta = kwargs.pop("eta", 1.0)
        super().__init__(*args, **kwargs)

    def _l_operator(self, qout):
        return qout.compute_expectation(self.eta)


class DPPL(AbstractDPP):
    """Dynamic Policy Programming with L operator.

    Args:
      eta (float): Positive constant.

    For other arguments, see DQN.
    """

    def __init__(self, *args, **kwargs):
        self.eta = kwargs.pop("eta", 1.0)
        super().__init__(*args, **kwargs)

    def _l_operator(self, qout):
        return torch.logsumexp(self.eta * qout.q_values, dim=1) / self.eta


class DPPGreedy(AbstractDPP):
    """Dynamic Policy Programming with max operator.

    This algorithm corresponds to DPP with eta = infinity.
    """

    def _l_operator(self, qout):
        return qout.max
