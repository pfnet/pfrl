from logging import getLogger

import numpy as np
import torch
from torch.nn import functional as F

import pfrl
from pfrl.agents import GoalConditionedTD3
from pfrl.utils.batch_states import batch_states
from pfrl.utils import clip_l2_grad_norm_


def default_target_policy_smoothing_func(batch_action):
    """Add noises to actions for target policy smoothing."""
    noise = torch.clamp(0.2 * torch.randn_like(batch_action), -0.5, 0.5)
    return torch.clamp(batch_action + noise, -1, 1)


class HIROHighLevelGoalConditionedTD3(GoalConditionedTD3):
    """
    HIRO Goal conditioned (including support for high and low level controllers)
    Twin Delayed Deep Deterministic Policy Gradients (TD3).

    See http://arxiv.org/abs/1802.09477

    Args:
        policy (Policy): Policy.
        q_func1 (Module): First Q-function that takes state-action pairs as input
            and outputs predicted Q-values.
        q_func2 (Module): Second Q-function that takes state-action pairs as
            input and outputs predicted Q-values.
        policy_optimizer (Optimizer): Optimizer setup with the policy
        q_func1_optimizer (Optimizer): Optimizer setup with the first
            Q-function.
        q_func2_optimizer (Optimizer): Optimizer setup with the second
            Q-function.
        replay_buffer (ReplayBuffer): Replay buffer
        gamma (float): Discount factor
        explorer (Explorer): Explorer that specifies an exploration strategy.
        gpu (int): GPU device id if not None nor negative.
        replay_start_size (int): if the replay buffer's size is less than
            replay_start_size, skip update
        minibatch_size (int): Minibatch size
        update_interval (int): Model update interval in step
        phi (callable): Feature extractor applied to observations
        soft_update_tau (float): Tau of soft target update.
        logger (Logger): Logger used
        batch_states (callable): method which makes a batch of observations.
            default is `pfrl.utils.batch_states.batch_states`
        burnin_action_func (callable or None): If not None, this callable
            object is used to select actions before the model is updated
            one or more times during training.
        policy_update_delay (int): Delay of policy updates. Policy is updated
            once in `policy_update_delay` times of Q-function updates.
        target_policy_smoothing_func (callable): Callable that takes a batch of
            actions as input and outputs a noisy version of it. It is used for
            target policy smoothing when computing target Q-values.
    """

    saved_attributes = (
        "policy",
        "q_func1",
        "q_func2",
        "target_policy",
        "target_q_func1",
        "target_q_func2",
        "policy_optimizer",
        "q_func1_optimizer",
        "q_func2_optimizer",
    )

    def __init__(
        self,
        policy,
        q_func1,
        q_func2,
        policy_optimizer,
        q_func1_optimizer,
        q_func2_optimizer,
        replay_buffer,
        gamma,
        explorer,
        gpu=None,
        replay_start_size=10000,
        minibatch_size=100,
        update_interval=1,
        phi=lambda x: x,
        soft_update_tau=5e-3,
        n_times_update=1,
        max_grad_norm=None,
        logger=getLogger(__name__),
        batch_states=batch_states,
        burnin_action_func=None,
        policy_update_delay=2,
        buffer_freq=10,
        q_func_grad_variance_record_size=10,
        policy_grad_variance_record_size=100,
        recent_variance_size=100,
        target_policy_smoothing_func=default_target_policy_smoothing_func,
        add_entropy=False
    ):
        # determines if we're dealing with a low level controller.
        self.cumulative_reward = False
        super(HIROHighLevelGoalConditionedTD3, self).__init__(policy=policy,
                                                              q_func1=q_func1,
                                                              q_func2=q_func2,
                                                              policy_optimizer=policy_optimizer,
                                                              q_func1_optimizer=q_func1_optimizer,
                                                              q_func2_optimizer=q_func2_optimizer,
                                                              replay_buffer=replay_buffer,
                                                              gamma=gamma,
                                                              explorer=explorer,
                                                              gpu=gpu,
                                                              replay_start_size=replay_start_size,
                                                              minibatch_size=minibatch_size,
                                                              update_interval=update_interval,
                                                              phi=phi,
                                                              soft_update_tau=soft_update_tau,
                                                              n_times_update=n_times_update,
                                                              max_grad_norm=max_grad_norm,
                                                              logger=logger,
                                                              batch_states=batch_states,
                                                              buffer_freq=buffer_freq,
                                                              burnin_action_func=burnin_action_func,
                                                              policy_update_delay=policy_update_delay,
                                                              q_func_grad_variance_record_size=q_func_grad_variance_record_size,
                                                              policy_grad_variance_record_size=policy_grad_variance_record_size,
                                                              recent_variance_size=recent_variance_size,
                                                              target_policy_smoothing_func=target_policy_smoothing_func,
                                                              add_entropy=add_entropy)

    def change_temporal_delay(self, new_temporal_delay):
        self.buffer_freq = new_temporal_delay

    def update_high_level_last_results(self, states, goals, actions):
        """
        update the last observation, goal and action for the high level
        controller.
        """
        self.batch_last_obs = [states]
        self.batch_last_goal = [goals]
        self.batch_last_action = [actions]

    def high_level_update_q_func_with_goal(self, batch):
        """
        Compute loss for a given Q-function, or critics
        for the high level controller
        """

        batch_next_state = batch["next_state"]
        batch_rewards = batch["reward"]
        batch_terminal = batch["is_state_terminal"]
        batch_state = batch["state"]
        batch_goal = batch["goal"]
        batch_actions = batch["action"]
        batch_discount = batch["discount"]

        with torch.no_grad(), pfrl.utils.evaluating(
            self.target_policy
        ), pfrl.utils.evaluating(self.target_q_func1), pfrl.utils.evaluating(
            self.target_q_func2
        ):
            next_action_distrib = self.target_policy(torch.cat([batch_next_state, batch_goal], -1))
            next_actions = self.target_policy_smoothing_func(
                next_action_distrib.sample()
            )

            entropy_term = 0
            if self.add_entropy:
                next_log_prob = next_action_distrib.log_prob(next_actions)
                entropy_term = self.temperature * next_log_prob[..., None]

            next_q1 = self.target_q_func1((torch.cat([batch_next_state, batch_goal], -1), next_actions))
            next_q2 = self.target_q_func2((torch.cat([batch_next_state, batch_goal], -1), next_actions))
            next_q = torch.min(next_q1, next_q2)

            target_q = batch_rewards + batch_discount * (
                1.0 - batch_terminal
            ) * torch.flatten(next_q - entropy_term)

        predict_q1 = torch.flatten(self.q_func1((torch.cat([batch_state, batch_goal], -1), batch_actions)))
        predict_q2 = torch.flatten(self.q_func2((torch.cat([batch_state, batch_goal], -1), batch_actions)))

        loss1 = F.smooth_l1_loss(target_q, predict_q1)
        loss2 = F.smooth_l1_loss(target_q, predict_q2)

        # Update stats
        self.q1_record.extend(predict_q1.detach().cpu().numpy())
        self.q2_record.extend(predict_q2.detach().cpu().numpy())
        self.q_func1_loss_record.append(float(loss1))
        self.q_func2_loss_record.append(float(loss2))

        q1_recent_variance = np.var(list(self.q1_record)[-self.recent_variance_size:])
        q2_recent_variance = np.var(list(self.q2_record)[-self.recent_variance_size:])
        self.q_func1_variance_record.append(q1_recent_variance)
        self.q_func2_variance_record.append(q2_recent_variance)

        self.q_func1_optimizer.zero_grad()
        loss1.backward()
        if self.max_grad_norm is not None:
            clip_l2_grad_norm_(self.q_func1.parameters(), self.max_grad_norm)
        self.q_func1_optimizer.step()

        self.q_func2_optimizer.zero_grad()
        loss2.backward()
        if self.max_grad_norm is not None:
            clip_l2_grad_norm_(self.q_func2.parameters(), self.max_grad_norm)
        self.q_func2_optimizer.step()

        self.q_func_n_updates += 1

    def high_level_update_batch(self, batch, errors_out=None):
        """Update the model from *batched*
           experiences for the high level controller

           This behavior is needed due to the extra machinery for the off policy
           correction.
        """
        # dealing with high level controller

        self.high_level_update_q_func_with_goal(batch)
        if self.q_func_n_updates % self.policy_update_delay == 0:
            self.update_policy_with_goal(batch)
            self.sync_target_network()

    def batch_observe_with_goal_state_action_arr(self, state_arr, action_arr, batch_obs, batch_goal, batch_reward, batch_done, batch_reset):
        if self.training:
            self._batch_observe_train_goal(batch_obs, batch_goal, batch_reward, batch_done, batch_reset,
                                           state_arr=state_arr, action_arr=action_arr)

    def _batch_observe_train_goal(self, batch_obs, batch_goal, batch_reward, batch_done, batch_reset, state_arr=None, action_arr=None):
        assert self.training
        if not self.cumulative_reward:
            self.cumulative_reward = np.zeros(len(batch_obs))
        for i in range(len(batch_obs)):
            self.t += 1
            if self.batch_last_obs[i] is not None:
                assert self.batch_last_goal[i] is not None
                assert self.batch_last_action[i] is not None
                # Add a transition to the replay buffer
                # high level controller, called every 10 times in
                # the hiro paper.
                arrs_exist = (state_arr is not None) and (action_arr is not None)
                if len(state_arr) == self.buffer_freq and arrs_exist:
                    equal_vals = self.batch_last_goal[i] == batch_goal[i]
                    if equal_vals.sum() != len(batch_goal[i]):
                        raise ValueError("Different values for final goal!")
                    self.cumulative_reward[i] = batch_reward[i]
                    self.replay_buffer.append(
                        state=self.batch_last_obs[i],
                        goal=self.batch_last_goal[i],
                        action=self.batch_last_action[i],
                        reward=self.cumulative_reward[i],
                        next_state=batch_obs[i],
                        next_action=None,
                        is_state_terminal=batch_done[i],
                        state_arr=state_arr,
                        action_arr=action_arr,
                        env_id=i
                    )
                    self.cumulative_reward = np.zeros(len(batch_obs))

                if batch_reset[i] or batch_done[i]:
                    self.batch_last_obs[i] = None
                    self.batch_last_goal[i] = None
                    self.batch_last_action[i] = None
                    self.replay_buffer.stop_current_episode(env_id=i)
