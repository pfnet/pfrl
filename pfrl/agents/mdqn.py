import collections
from logging import getLogger

import torch
import torch.nn.functional as F
import numpy as np

from pfrl.agents import dqn
from pfrl.utils.batch_states import batch_states
from pfrl.utils.recurrent import pack_and_forward


def _mean_or_nan(xs):
    """Return its mean a non-empty sequence, numpy.nan for a empty one."""
    return np.mean(xs) if xs else np.nan


class MDQN(dqn.DQN):
    """Munchausen Deep Q-Network algorithm.

    See https://arxiv.org/abs/2007.14430.

    Args:
        q_function (StateQFunction): Q-function
        optimizer (Optimizer): Optimizer that is already setup
        replay_buffer (ReplayBuffer): Replay buffer
        gamma (float): Discount factor
        explorer (Explorer): Explorer that specifies an exploration strategy.
        gpu (int): GPU device id if not None nor negative.
        replay_start_size (int): if the replay buffer's size is less than
            replay_start_size, skip update
        minibatch_size (int): Minibatch size
        update_interval (int): Model update interval in step
        target_update_interval (int): Target model update interval in step
        clip_delta (bool): Clip delta if set True
        phi (callable): Feature extractor applied to observations
        target_update_method (str): 'hard' or 'soft'.
        soft_update_tau (float): Tau of soft target update.
        n_times_update (int): Number of repetition of update
        batch_accumulator (str): 'mean' or 'sum'
        episodic_update_len (int or None): Subsequences of this length are used
            for update if set int and episodic_update=True
        logger (Logger): Logger used
        batch_states (callable): method which makes a batch of observations.
            default is `pfrl.utils.batch_states.batch_states`
        recurrent (bool): If set to True, `model` is assumed to implement
            `pfrl.nn.Recurrent` and is updated in a recurrent
            manner.
        max_grad_norm (float or None): Maximum L2 norm of the gradient used for
            gradient clipping. If set to None, the gradient is not clipped.
        temperature (float): entropy temperature
        scaling_term (float): Munchausen scaling term
        clip_l0 (float): log-policy clipping value
    """

    saved_attributes = ("model", "target_model", "optimizer")

    def __init__(
        self,
        q_function,
        optimizer,
        replay_buffer,
        gamma,
        explorer,
        gpu=None,
        replay_start_size=50000,
        minibatch_size=32,
        update_interval=1,
        target_update_interval=10000,
        clip_delta=True,
        phi=lambda x: x,
        target_update_method="hard",
        soft_update_tau=1e-2,
        n_times_update=1,
        batch_accumulator="mean",
        episodic_update_len=None,
        logger=getLogger(__name__),
        batch_states=batch_states,
        recurrent=False,
        max_grad_norm=None,
        temperature=0.03,
        scaling_term=0.9,
        clip_l0=-1.0,
    ):
        super(MDQN, self).__init__(
            q_function,
            optimizer,
            replay_buffer,
            gamma,
            explorer,
            gpu=gpu,
            replay_start_size=replay_start_size,
            minibatch_size=minibatch_size,
            update_interval=update_interval,
            target_update_interval=target_update_interval,
            clip_delta=clip_delta,
            phi=phi,
            target_update_method=target_update_method,
            soft_update_tau=soft_update_tau,
            n_times_update=n_times_update,
            batch_accumulator=batch_accumulator,
            episodic_update_len=episodic_update_len,
            logger=logger,
            batch_states=batch_states,
            recurrent=recurrent,
            max_grad_norm=max_grad_norm,
        )

        self.temperature = temperature
        self.scaling_term = scaling_term
        self.clip_l0 = clip_l0

        self.pi_sum_record = collections.deque(maxlen=1000)
        self.chosen_pi_record = collections.deque(maxlen=1000)
        self.bonus_reward_record = collections.deque(maxlen=1000)
        self.augmented_reward_record = collections.deque(maxlen=1000)
        self.next_pi_sum_record = collections.deque(maxlen=1000)
        self.next_value_record = collections.deque(maxlen=1000)
        self.next_entropy_record = collections.deque(maxlen=1000)

    def _compute_target_values(self, exp_batch):
        # Compute Q-values for current states using the target network
        batch_state = exp_batch["state"]

        if self.recurrent:
            qout, _ = pack_and_forward(
                self.target_model, batch_state, exp_batch["recurrent_state"]
            )
        else:
            qout = self.target_model(batch_state)

        # log-sum-exp-trick
        advantages = qout.q_values - qout.max.unsqueeze(1)
        t_ln_pi = advantages - self.temperature * (
            advantages / self.temperature
        ).logsumexp(dim=1, keepdim=True)
        pi = (t_ln_pi / self.temperature).exp()
        self.pi_sum_record.extend(pi.sum(dim=1).detach().cpu().numpy())

        # add scaled log policy
        batch_actions = exp_batch["action"].long().unsqueeze(1)
        chosen_t_ln_pi = t_ln_pi.gather(dim=1, index=batch_actions).flatten()
        chosen_pi = (chosen_t_ln_pi / self.temperature).exp()
        self.chosen_pi_record.extend(chosen_pi.detach().cpu().numpy())
        bonus = self.scaling_term * torch.clamp(chosen_t_ln_pi, min=self.clip_l0, max=0)
        self.bonus_reward_record.extend(bonus.detach().cpu().numpy())
        augmented_rewards = exp_batch["reward"] + bonus
        self.augmented_reward_record.extend(augmented_rewards.detach().cpu().numpy())

        # value of next state (entropy-augmented) using the target network
        batch_next_state = exp_batch["next_state"]

        if self.recurrent:
            target_next_qout, _ = pack_and_forward(
                self.target_model, batch_next_state, exp_batch["next_recurrent_state"],
            )
        else:
            target_next_qout = self.target_model(batch_next_state)
        next_q_max = target_next_qout.max

        # log-sum-exp-trick
        next_advantages = target_next_qout.q_values - next_q_max.unsqueeze(1)
        next_t_ln_pi = next_advantages - self.temperature * (
            next_advantages / self.temperature
        ).logsumexp(dim=1, keepdim=True)
        #next_pi = (next_t_ln_pi / self.temperature).exp()
        next_pi = F.softmax(target_next_qout.q_values / self.temperature, dim=1)
        #next_pi = F.softmax(target_next_qout.q_values, dim=1)
        self.next_pi_sum_record.extend(next_pi.sum(dim=1).detach().cpu().numpy())
        next_value = (next_pi * (target_next_qout.q_values - next_t_ln_pi)).sum(dim=1)
        self.next_value_record.extend(next_value.detach().cpu().numpy())

        next_entropy = -(next_pi * next_t_ln_pi).sum(dim=1)
        self.next_entropy_record.extend(next_entropy.detach().cpu().numpy())

        batch_terminal = exp_batch["is_state_terminal"]
        discount = exp_batch["discount"]

        return augmented_rewards + discount * (1.0 - batch_terminal) * next_value

    def get_statistics(self):
        return super(MDQN, self).get_statistics() + [
            ("chosen_pi", _mean_or_nan(self.chosen_pi_record)),
            ("pi_sum", _mean_or_nan(self.pi_sum_record)),
            ("bonus", _mean_or_nan(self.bonus_reward_record)),
            ("augmented_reward", _mean_or_nan(self.augmented_reward_record)),
            ("next_pi_sum", _mean_or_nan(self.next_pi_sum_record)),
            ("next_value", _mean_or_nan(self.next_value_record)),
            ("next_entropy", _mean_or_nan(self.next_entropy_record)),
        ]
