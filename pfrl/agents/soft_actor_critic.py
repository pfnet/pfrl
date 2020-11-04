import collections
import copy
from logging import getLogger

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import pfrl
from pfrl.agent import AttributeSavingMixin, BatchAgent
from pfrl.replay_buffer import ReplayUpdater, batch_experiences
from pfrl.utils import clip_l2_grad_norm_
from pfrl.utils.batch_states import batch_states
from pfrl.utils.copy_param import synchronize_parameters
from pfrl.utils.mode_of_distribution import mode_of_distribution


def _mean_or_nan(xs):
    """Return its mean a non-empty sequence, numpy.nan for a empty one."""
    return np.mean(xs) if xs else np.nan


class TemperatureHolder(nn.Module):
    """Module that holds a temperature as a learnable value.

    Args:
        initial_log_temperature (float): Initial value of log(temperature).
    """

    def __init__(self, initial_log_temperature=0):
        super().__init__()
        self.log_temperature = nn.Parameter(
            torch.tensor(initial_log_temperature, dtype=torch.float32)
        )

    def forward(self):
        """Return a temperature as a torch.Tensor."""
        return torch.exp(self.log_temperature)


class SoftActorCritic(AttributeSavingMixin, BatchAgent):
    """Soft Actor-Critic (SAC).

    See https://arxiv.org/abs/1812.05905

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
        initial_temperature (float): Initial temperature value. If
            `entropy_target` is set to None, the temperature is fixed to it.
        entropy_target (float or None): If set to a float, the temperature is
            adjusted during training to match the policy's entropy to it.
        temperature_optimizer_lr (float): Learning rate of the temperature
            optimizer. If set to None, Adam with default hyperparameters
            is used.
        act_deterministically (bool): If set to True, choose most probable
            actions in the act method instead of sampling from distributions.
    """

    saved_attributes = (
        "policy",
        "q_func1",
        "q_func2",
        "target_q_func1",
        "target_q_func2",
        "policy_optimizer",
        "q_func1_optimizer",
        "q_func2_optimizer",
        "temperature_holder",
        "temperature_optimizer",
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
        gpu=None,
        replay_start_size=10000,
        minibatch_size=100,
        update_interval=1,
        phi=lambda x: x,
        soft_update_tau=5e-3,
        max_grad_norm=None,
        logger=getLogger(__name__),
        batch_states=batch_states,
        burnin_action_func=None,
        initial_temperature=1.0,
        entropy_target=None,
        temperature_optimizer_lr=None,
        act_deterministically=True,
    ):

        self.policy = policy
        self.q_func1 = q_func1
        self.q_func2 = q_func2

        if gpu is not None and gpu >= 0:
            assert torch.cuda.is_available()
            self.device = torch.device("cuda:{}".format(gpu))
            self.policy.to(self.device)
            self.q_func1.to(self.device)
            self.q_func2.to(self.device)
        else:
            self.device = torch.device("cpu")

        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.gpu = gpu
        self.phi = phi
        self.soft_update_tau = soft_update_tau
        self.logger = logger
        self.policy_optimizer = policy_optimizer
        self.q_func1_optimizer = q_func1_optimizer
        self.q_func2_optimizer = q_func2_optimizer
        self.replay_updater = ReplayUpdater(
            replay_buffer=replay_buffer,
            update_func=self.update,
            batchsize=minibatch_size,
            n_times_update=1,
            replay_start_size=replay_start_size,
            update_interval=update_interval,
            episodic_update=False,
        )
        self.max_grad_norm = max_grad_norm
        self.batch_states = batch_states
        self.burnin_action_func = burnin_action_func
        self.initial_temperature = initial_temperature
        self.entropy_target = entropy_target
        if self.entropy_target is not None:
            self.temperature_holder = TemperatureHolder(
                initial_log_temperature=np.log(initial_temperature)
            )
            if temperature_optimizer_lr is not None:
                self.temperature_optimizer = torch.optim.Adam(
                    self.temperature_holder.parameters(), lr=temperature_optimizer_lr
                )
            else:
                self.temperature_optimizer = torch.optim.Adam(
                    self.temperature_holder.parameters()
                )
            if gpu is not None and gpu >= 0:
                self.temperature_holder.to(self.device)
        else:
            self.temperature_holder = None
            self.temperature_optimizer = None
        self.act_deterministically = act_deterministically

        self.t = 0

        # Target model
        self.target_q_func1 = copy.deepcopy(self.q_func1).eval().requires_grad_(False)
        self.target_q_func2 = copy.deepcopy(self.q_func2).eval().requires_grad_(False)

        # Statistics
        self.q1_record = collections.deque(maxlen=1000)
        self.q2_record = collections.deque(maxlen=1000)
        self.entropy_record = collections.deque(maxlen=1000)
        self.q_func1_loss_record = collections.deque(maxlen=100)
        self.q_func2_loss_record = collections.deque(maxlen=100)
        self.n_policy_updates = 0

    @property
    def temperature(self):
        if self.entropy_target is None:
            return self.initial_temperature
        else:
            with torch.no_grad():
                return float(self.temperature_holder())

    def sync_target_network(self):
        """Synchronize target network with current network."""
        synchronize_parameters(
            src=self.q_func1,
            dst=self.target_q_func1,
            method="soft",
            tau=self.soft_update_tau,
        )
        synchronize_parameters(
            src=self.q_func2,
            dst=self.target_q_func2,
            method="soft",
            tau=self.soft_update_tau,
        )

    def update_q_func(self, batch):
        """Compute loss for a given Q-function."""

        batch_next_state = batch["next_state"]
        batch_rewards = batch["reward"]
        batch_terminal = batch["is_state_terminal"]
        batch_state = batch["state"]
        batch_actions = batch["action"]
        batch_discount = batch["discount"]

        with torch.no_grad(), pfrl.utils.evaluating(self.policy), pfrl.utils.evaluating(
            self.target_q_func1
        ), pfrl.utils.evaluating(self.target_q_func2):
            next_action_distrib = self.policy(batch_next_state)
            next_actions = next_action_distrib.sample()
            next_log_prob = next_action_distrib.log_prob(next_actions)
            next_q1 = self.target_q_func1((batch_next_state, next_actions))
            next_q2 = self.target_q_func2((batch_next_state, next_actions))
            next_q = torch.min(next_q1, next_q2)
            entropy_term = self.temperature * next_log_prob[..., None]
            assert next_q.shape == entropy_term.shape

            target_q = batch_rewards + batch_discount * (
                1.0 - batch_terminal
            ) * torch.flatten(next_q - entropy_term)

        predict_q1 = torch.flatten(self.q_func1((batch_state, batch_actions)))
        predict_q2 = torch.flatten(self.q_func2((batch_state, batch_actions)))

        loss1 = 0.5 * F.mse_loss(target_q, predict_q1)
        loss2 = 0.5 * F.mse_loss(target_q, predict_q2)

        # Update stats
        self.q1_record.extend(predict_q1.detach().cpu().numpy())
        self.q2_record.extend(predict_q2.detach().cpu().numpy())
        self.q_func1_loss_record.append(float(loss1))
        self.q_func2_loss_record.append(float(loss2))

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

    def update_temperature(self, log_prob):
        assert not log_prob.requires_grad
        loss = -torch.mean(self.temperature_holder() * (log_prob + self.entropy_target))
        self.temperature_optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            clip_l2_grad_norm_(self.temperature_holder.parameters(), self.max_grad_norm)
        self.temperature_optimizer.step()

    def update_policy_and_temperature(self, batch):
        """Compute loss for actor."""

        batch_state = batch["state"]

        action_distrib = self.policy(batch_state)
        actions = action_distrib.rsample()
        log_prob = action_distrib.log_prob(actions)
        q1 = self.q_func1((batch_state, actions))
        q2 = self.q_func2((batch_state, actions))
        q = torch.min(q1, q2)

        entropy_term = self.temperature * log_prob[..., None]
        assert q.shape == entropy_term.shape
        loss = torch.mean(entropy_term - q)

        self.policy_optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            clip_l2_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()

        self.n_policy_updates += 1

        if self.entropy_target is not None:
            self.update_temperature(log_prob.detach())

        # Record entropy
        with torch.no_grad():
            try:
                self.entropy_record.extend(
                    action_distrib.entropy().detach().cpu().numpy()
                )
            except NotImplementedError:
                # Record - log p(x) instead
                self.entropy_record.extend(-log_prob.detach().cpu().numpy())

    def update(self, experiences, errors_out=None):
        """Update the model from experiences"""
        batch = batch_experiences(experiences, self.device, self.phi, self.gamma)
        self.update_q_func(batch)
        self.update_policy_and_temperature(batch)
        self.sync_target_network()

    def batch_select_greedy_action(self, batch_obs, deterministic=False):
        with torch.no_grad(), pfrl.utils.evaluating(self.policy):
            batch_xs = self.batch_states(batch_obs, self.device, self.phi)
            policy_out = self.policy(batch_xs)
            if deterministic:
                batch_action = mode_of_distribution(policy_out).cpu().numpy()
            else:
                batch_action = policy_out.sample().cpu().numpy()
        return batch_action

    def batch_act(self, batch_obs):
        if self.training:
            return self._batch_act_train(batch_obs)
        else:
            return self._batch_act_eval(batch_obs)

    def batch_observe(self, batch_obs, batch_reward, batch_done, batch_reset):
        if self.training:
            self._batch_observe_train(batch_obs, batch_reward, batch_done, batch_reset)

    def _batch_act_eval(self, batch_obs):
        assert not self.training
        return self.batch_select_greedy_action(
            batch_obs, deterministic=self.act_deterministically
        )

    def _batch_act_train(self, batch_obs):
        assert self.training
        if self.burnin_action_func is not None and self.n_policy_updates == 0:
            batch_action = [self.burnin_action_func() for _ in range(len(batch_obs))]
        else:
            batch_action = self.batch_select_greedy_action(batch_obs)
        self.batch_last_obs = list(batch_obs)
        self.batch_last_action = list(batch_action)

        return batch_action

    def _batch_observe_train(self, batch_obs, batch_reward, batch_done, batch_reset):
        assert self.training
        for i in range(len(batch_obs)):
            self.t += 1
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
            ("average_q1", _mean_or_nan(self.q1_record)),
            ("average_q2", _mean_or_nan(self.q2_record)),
            ("average_q_func1_loss", _mean_or_nan(self.q_func1_loss_record)),
            ("average_q_func2_loss", _mean_or_nan(self.q_func2_loss_record)),
            ("n_updates", self.n_policy_updates),
            ("average_entropy", _mean_or_nan(self.entropy_record)),
            ("temperature", self.temperature),
        ]
