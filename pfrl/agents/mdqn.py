from logging import getLogger

import torch

from pfrl.agents import dqn
from pfrl.utils.batch_states import batch_states
from pfrl.utils.recurrent import pack_and_forward


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
        ).exp().sum(dim=1).log().unsqueeze(1)
        pi = (t_ln_pi / self.temperature).exp()

        # add scaled log policy
        batch_actions = exp_batch["action"].long().unsqueeze(1)
        chosen_t_ln_pi = t_ln_pi.gather(dim=1, index=batch_actions).flatten()
        exp_batch["reward"] += self.scaling_term * torch.max(
            chosen_t_ln_pi, torch.tensor(self.clip_l0, device=self.device)
        )

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
            advantages / self.temperature
        ).exp().sum(dim=1).log().unsqueeze(1)
        next_value = torch.sum(pi * (target_next_qout.q_values - next_t_ln_pi), dim=1)

        batch_rewards = exp_batch["reward"]
        batch_terminal = exp_batch["is_state_terminal"]
        discount = exp_batch["discount"]

        return batch_rewards + discount * (1.0 - batch_terminal) * next_value
