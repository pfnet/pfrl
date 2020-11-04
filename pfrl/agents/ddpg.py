import collections
import copy
from logging import getLogger

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pfrl.agent import AttributeSavingMixin, BatchAgent
from pfrl.replay_buffer import ReplayUpdater, batch_experiences
from pfrl.utils.batch_states import batch_states
from pfrl.utils.contexts import evaluating
from pfrl.utils.copy_param import synchronize_parameters


def _mean_or_nan(xs):
    """Return its mean a non-empty sequence, numpy.nan for a empty one."""
    return np.mean(xs) if xs else np.nan


class DDPG(AttributeSavingMixin, BatchAgent):
    """Deep Deterministic Policy Gradients.

    This can be used as SVG(0) by specifying a Gaussian policy instead of a
    deterministic policy.

    Args:
        policy (torch.nn.Module): Policy
        q_func (torch.nn.Module): Q-function
        actor_optimizer (Optimizer): Optimizer setup with the policy
        critic_optimizer (Optimizer): Optimizer setup with the Q-function
        replay_buffer (ReplayBuffer): Replay buffer
        gamma (float): Discount factor
        explorer (Explorer): Explorer that specifies an exploration strategy.
        gpu (int): GPU device id if not None nor negative.
        replay_start_size (int): if the replay buffer's size is less than
            replay_start_size, skip update
        minibatch_size (int): Minibatch size
        update_interval (int): Model update interval in step
        target_update_interval (int): Target model update interval in step
        phi (callable): Feature extractor applied to observations
        target_update_method (str): 'hard' or 'soft'.
        soft_update_tau (float): Tau of soft target update.
        n_times_update (int): Number of repetition of update
        batch_accumulator (str): 'mean' or 'sum'
        episodic_update (bool): Use full episodes for update if set True
        episodic_update_len (int or None): Subsequences of this length are used
            for update if set int and episodic_update=True
        logger (Logger): Logger used
        batch_states (callable): method which makes a batch of observations.
            default is `pfrl.utils.batch_states.batch_states`
        burnin_action_func (callable or None): If not None, this callable
            object is used to select actions before the model is updated
            one or more times during training.
    """

    saved_attributes = ("model", "target_model", "actor_optimizer", "critic_optimizer")

    def __init__(
        self,
        policy,
        q_func,
        actor_optimizer,
        critic_optimizer,
        replay_buffer,
        gamma,
        explorer,
        gpu=None,
        replay_start_size=50000,
        minibatch_size=32,
        update_interval=1,
        target_update_interval=10000,
        phi=lambda x: x,
        target_update_method="hard",
        soft_update_tau=1e-2,
        n_times_update=1,
        recurrent=False,
        episodic_update_len=None,
        logger=getLogger(__name__),
        batch_states=batch_states,
        burnin_action_func=None,
    ):

        self.model = nn.ModuleList([policy, q_func])
        if gpu is not None and gpu >= 0:
            assert torch.cuda.is_available()
            self.device = torch.device("cuda:{}".format(gpu))
            self.model.to(self.device)
        else:
            self.device = torch.device("cpu")

        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.explorer = explorer
        self.gpu = gpu
        self.target_update_interval = target_update_interval
        self.phi = phi
        self.target_update_method = target_update_method
        self.soft_update_tau = soft_update_tau
        self.logger = logger
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.recurrent = recurrent
        assert not self.recurrent, "recurrent=True is not yet implemented"
        if self.recurrent:
            update_func = self.update_from_episodes
        else:
            update_func = self.update
        self.replay_updater = ReplayUpdater(
            replay_buffer=replay_buffer,
            update_func=update_func,
            batchsize=minibatch_size,
            episodic_update=recurrent,
            episodic_update_len=episodic_update_len,
            n_times_update=n_times_update,
            replay_start_size=replay_start_size,
            update_interval=update_interval,
        )
        self.batch_states = batch_states
        self.burnin_action_func = burnin_action_func

        self.t = 0
        self.last_state = None
        self.last_action = None
        self.target_model = copy.deepcopy(self.model)
        self.target_model.eval()
        self.q_record = collections.deque(maxlen=1000)
        self.actor_loss_record = collections.deque(maxlen=100)
        self.critic_loss_record = collections.deque(maxlen=100)
        self.n_updates = 0

        # Aliases for convenience
        self.policy, self.q_function = self.model
        self.target_policy, self.target_q_function = self.target_model

        self.sync_target_network()

    def sync_target_network(self):
        """Synchronize target network with current network."""
        synchronize_parameters(
            src=self.model,
            dst=self.target_model,
            method=self.target_update_method,
            tau=self.soft_update_tau,
        )

    # Update Q-function
    def compute_critic_loss(self, batch):
        """Compute loss for critic."""

        batch_next_state = batch["next_state"]
        batch_rewards = batch["reward"]
        batch_terminal = batch["is_state_terminal"]
        batch_state = batch["state"]
        batch_actions = batch["action"]
        batchsize = len(batch_rewards)

        with torch.no_grad():
            assert not self.recurrent
            next_actions = self.target_policy(batch_next_state).sample()
            next_q = self.target_q_function((batch_next_state, next_actions))
            target_q = batch_rewards + self.gamma * (
                1.0 - batch_terminal
            ) * next_q.reshape((batchsize,))

        predict_q = self.q_function((batch_state, batch_actions)).reshape((batchsize,))

        loss = F.mse_loss(target_q, predict_q)

        # Update stats
        self.critic_loss_record.append(float(loss.detach().cpu().numpy()))

        return loss

    def compute_actor_loss(self, batch):
        """Compute loss for actor."""

        batch_state = batch["state"]
        onpolicy_actions = self.policy(batch_state).rsample()
        q = self.q_function((batch_state, onpolicy_actions))
        loss = -q.mean()

        # Update stats
        self.q_record.extend(q.detach().cpu().numpy())
        self.actor_loss_record.append(float(loss.detach().cpu().numpy()))

        return loss

    def update(self, experiences, errors_out=None):
        """Update the model from experiences"""

        batch = batch_experiences(experiences, self.device, self.phi, self.gamma)

        self.critic_optimizer.zero_grad()
        self.compute_critic_loss(batch).backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        self.compute_actor_loss(batch).backward()
        self.actor_optimizer.step()

        self.n_updates += 1

    def update_from_episodes(self, episodes, errors_out=None):
        raise NotImplementedError

        # Sort episodes desc by their lengths
        sorted_episodes = list(reversed(sorted(episodes, key=len)))
        max_epi_len = len(sorted_episodes[0])

        # Precompute all the input batches
        batches = []
        for i in range(max_epi_len):
            transitions = []
            for ep in sorted_episodes:
                if len(ep) <= i:
                    break
                transitions.append([ep[i]])
            batch = batch_experiences(
                transitions, xp=self.device, phi=self.phi, gamma=self.gamma
            )
            batches.append(batch)

        with self.model.state_reset(), self.target_model.state_reset():

            # Since the target model is evaluated one-step ahead,
            # its internal states need to be updated
            self.target_q_function.update_state(
                batches[0]["state"], batches[0]["action"]
            )
            self.target_policy(batches[0]["state"])

            # Update critic through time
            critic_loss = 0
            for batch in batches:
                critic_loss += self.compute_critic_loss(batch)
            self.critic_optimizer.update(lambda: critic_loss / max_epi_len)

        with self.model.state_reset():

            # Update actor through time
            actor_loss = 0
            for batch in batches:
                actor_loss += self.compute_actor_loss(batch)
            self.actor_optimizer.update(lambda: actor_loss / max_epi_len)

    def batch_act(self, batch_obs):
        if self.training:
            return self._batch_act_train(batch_obs)
        else:
            return self._batch_act_eval(batch_obs)

    def batch_observe(self, batch_obs, batch_reward, batch_done, batch_reset):
        if self.training:
            self._batch_observe_train(batch_obs, batch_reward, batch_done, batch_reset)

    def _batch_select_greedy_actions(self, batch_obs):
        with torch.no_grad(), evaluating(self.policy):
            batch_xs = self.batch_states(batch_obs, self.device, self.phi)
            batch_action = self.policy(batch_xs).sample()
            return batch_action.cpu().numpy()

    def _batch_act_eval(self, batch_obs):
        assert not self.training
        return self._batch_select_greedy_actions(batch_obs)

    def _batch_act_train(self, batch_obs):
        assert self.training
        if self.burnin_action_func is not None and self.n_updates == 0:
            batch_action = [self.burnin_action_func() for _ in range(len(batch_obs))]
        else:
            batch_greedy_action = self._batch_select_greedy_actions(batch_obs)
            batch_action = [
                self.explorer.select_action(self.t, lambda: batch_greedy_action[i])
                for i in range(len(batch_greedy_action))
            ]

        self.batch_last_obs = list(batch_obs)
        self.batch_last_action = list(batch_action)

        return batch_action

    def _batch_observe_train(self, batch_obs, batch_reward, batch_done, batch_reset):
        assert self.training
        for i in range(len(batch_obs)):
            self.t += 1
            # Update the target network
            if self.t % self.target_update_interval == 0:
                self.sync_target_network()
            if self.batch_last_obs[i] is not None:
                assert self.batch_last_action[i] is not None
                # Add a transition to the replay buffer
                self.replay_buffer.append(
                    state=self.batch_last_obs[i],
                    action=self.batch_last_action[i],
                    reward=batch_reward[i],
                    next_state=batch_obs[i],
                    next_action=None,
                    is_state_terminal=batch_done[i],
                    env_id=i,
                )
                if batch_reset[i] or batch_done[i]:
                    self.batch_last_obs[i] = None
                    self.batch_last_action[i] = None
                    self.replay_buffer.stop_current_episode(env_id=i)
            self.replay_updater.update_if_necessary(self.t)

    def get_statistics(self):
        return [
            ("average_q", _mean_or_nan(self.q_record)),
            ("average_actor_loss", _mean_or_nan(self.actor_loss_record)),
            ("average_critic_loss", _mean_or_nan(self.critic_loss_record)),
            ("n_updates", self.n_updates),
        ]
