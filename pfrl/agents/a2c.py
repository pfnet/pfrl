import warnings
from logging import getLogger

import torch

from pfrl import agent
from pfrl.utils import clip_l2_grad_norm_
from pfrl.utils.batch_states import batch_states
from pfrl.utils.mode_of_distribution import mode_of_distribution

logger = getLogger(__name__)


class A2C(agent.AttributeSavingMixin, agent.BatchAgent):
    """A2C: Advantage Actor-Critic.

    A2C is a synchronous, deterministic variant of Asynchronous Advantage
        Actor Critic (A3C).

    See https://arxiv.org/abs/1708.05144

    Args:
        model (nn.Module): Model to train
        optimizer (torch.optim.Optimizer): optimizer used to train the model
        gamma (float): Discount factor [0,1]
        num_processes (int): The number of processes
        gpu (int): GPU device id if not None nor negative.
        update_steps (int): The number of update steps
        phi (callable): Feature extractor function
        pi_loss_coef (float): Weight coefficient for the loss of the policy
        v_loss_coef (float): Weight coefficient for the loss of the value
            function
        entropy_coeff (float): Weight coefficient for the loss of the entropy
        use_gae (bool): use generalized advantage estimation(GAE)
        tau (float): gae parameter
        act_deterministically (bool): If set true, choose most probable actions
            in act method.
        max_grad_norm (float or None): Maximum L2 norm of the gradient used for
            gradient clipping. If set to None, the gradient is not clipped.
        average_actor_loss_decay (float): Decay rate of average actor loss.
            Used only to record statistics.
        average_entropy_decay (float): Decay rate of average entropy. Used only
            to record statistics.
        average_value_decay (float): Decay rate of average value. Used only
            to record statistics.
        batch_states (callable): method which makes a batch of observations.
            default is `pfrl.utils.batch_states.batch_states`
    """

    process_idx = None
    saved_attributes = ("model", "optimizer")

    def __init__(
        self,
        model,
        optimizer,
        gamma,
        num_processes,
        gpu=None,
        update_steps=5,
        phi=lambda x: x,
        pi_loss_coef=1.0,
        v_loss_coef=0.5,
        entropy_coeff=0.01,
        use_gae=False,
        tau=0.95,
        act_deterministically=False,
        max_grad_norm=None,
        average_actor_loss_decay=0.999,
        average_entropy_decay=0.999,
        average_value_decay=0.999,
        batch_states=batch_states,
    ):
        self.model = model
        if gpu is not None and gpu >= 0:
            assert torch.cuda.is_available()
            self.device = torch.device("cuda:{}".format(gpu))
            self.model.to(self.device)
        else:
            self.device = torch.device("cpu")

        self.optimizer = optimizer

        self.update_steps = update_steps
        self.num_processes = num_processes

        self.gamma = gamma
        self.use_gae = use_gae
        self.tau = tau
        self.act_deterministically = act_deterministically
        self.max_grad_norm = max_grad_norm
        self.phi = phi
        self.pi_loss_coef = pi_loss_coef
        self.v_loss_coef = v_loss_coef
        self.entropy_coeff = entropy_coeff

        self.average_actor_loss_decay = average_actor_loss_decay
        self.average_value_decay = average_value_decay
        self.average_entropy_decay = average_entropy_decay
        self.batch_states = batch_states

        self.t = 0
        self.t_start = 0

        # Stats
        self.average_actor_loss = 0
        self.average_value = 0
        self.average_entropy = 0

    def _flush_storage(self, obs_shape, action):
        obs_shape = obs_shape[1:]
        action_shape = action.shape[1:]

        self.states = torch.zeros(
            self.update_steps + 1,
            self.num_processes,
            *obs_shape,
            device=self.device,
            dtype=torch.float
        )
        self.actions = torch.zeros(
            self.update_steps,
            self.num_processes,
            *action_shape,
            device=self.device,
            dtype=torch.float
        )
        self.rewards = torch.zeros(
            self.update_steps, self.num_processes, device=self.device, dtype=torch.float
        )
        self.value_preds = torch.zeros(
            self.update_steps + 1,
            self.num_processes,
            device=self.device,
            dtype=torch.float,
        )
        self.returns = torch.zeros(
            self.update_steps + 1,
            self.num_processes,
            device=self.device,
            dtype=torch.float,
        )
        self.masks = torch.ones(
            self.update_steps, self.num_processes, device=self.device, dtype=torch.float
        )

        self.obs_shape = obs_shape
        self.action_shape = action_shape

    def _compute_returns(self, next_value):
        if self.use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for i in reversed(range(self.update_steps)):
                delta = (
                    self.rewards[i]
                    + self.gamma * self.value_preds[i + 1] * self.masks[i]
                    - self.value_preds[i]
                )
                gae = delta + self.gamma * self.tau * self.masks[i] * gae
                self.returns[i] = gae + self.value_preds[i]
        else:
            self.returns[-1] = next_value
            for i in reversed(range(self.update_steps)):
                self.returns[i] = (
                    self.rewards[i] + self.gamma * self.returns[i + 1] * self.masks[i]
                )

    def update(self):
        with torch.no_grad():
            _, next_value = self.model(self.states[-1])
            next_value = next_value[:, 0]

        self._compute_returns(next_value)
        pout, values = self.model(self.states[:-1].reshape(-1, *self.obs_shape))

        actions = self.actions.reshape(-1, *self.action_shape)
        dist_entropy = pout.entropy().mean()
        action_log_probs = pout.log_prob(actions)

        values = values.reshape((self.update_steps, self.num_processes))
        action_log_probs = action_log_probs.reshape(
            (self.update_steps, self.num_processes)
        )
        advantages = self.returns[:-1] - values
        value_loss = (advantages * advantages).mean()
        action_loss = -(advantages.detach() * action_log_probs).mean()

        self.optimizer.zero_grad()

        (
            value_loss * self.v_loss_coef
            + action_loss * self.pi_loss_coef
            - dist_entropy * self.entropy_coeff
        ).backward()

        if self.max_grad_norm is not None:
            clip_l2_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.states[0] = self.states[-1]

        self.t_start = self.t

        # Update stats
        self.average_actor_loss += (1 - self.average_actor_loss_decay) * (
            float(action_loss) - self.average_actor_loss
        )
        self.average_value += (1 - self.average_value_decay) * (
            float(value_loss) - self.average_value
        )
        self.average_entropy += (1 - self.average_entropy_decay) * (
            float(dist_entropy) - self.average_entropy
        )

    def batch_act(self, batch_obs):
        if self.training:
            return self._batch_act_train(batch_obs)
        else:
            return self._batch_act_eval(batch_obs)

    def batch_observe(self, batch_obs, batch_reward, batch_done, batch_reset):
        if self.training:
            self._batch_observe_train(batch_obs, batch_reward, batch_done, batch_reset)

    def _batch_act_train(self, batch_obs):
        assert self.training

        statevar = self.batch_states(batch_obs, self.device, self.phi)

        if self.t == 0:
            with torch.no_grad():
                pout, _ = self.model(statevar)
                action = pout.sample()
            self._flush_storage(statevar.shape, action)

        self.states[self.t - self.t_start] = statevar

        if self.t - self.t_start == self.update_steps:
            self.update()

        with torch.no_grad():
            pout, value = self.model(statevar)
            action = pout.sample()

        self.actions[self.t - self.t_start] = action.reshape(-1, *self.action_shape)
        self.value_preds[self.t - self.t_start] = value[:, 0]

        return action.cpu().numpy()

    def _batch_act_eval(self, batch_obs):
        assert not self.training
        statevar = self.batch_states(batch_obs, self.device, self.phi)
        with torch.no_grad():
            pout, _ = self.model(statevar)
            if self.act_deterministically:
                action = mode_of_distribution(pout)
            else:
                action = pout.sample()
        return action.cpu().numpy()

    def _batch_observe_train(self, batch_obs, batch_reward, batch_done, batch_reset):
        assert self.training
        self.t += 1

        if any(batch_reset):
            warnings.warn(
                "A2C currently does not support resetting an env without reaching a"
                " terminal state during training. When receiving True in batch_reset,"
                " A2C considers it as True in batch_done instead."
            )  # NOQA
            batch_done = list(batch_done)
            for i, reset in enumerate(batch_reset):
                if reset:
                    batch_done[i] = True

        statevar = self.batch_states(batch_obs, self.device, self.phi)

        self.masks[self.t - self.t_start - 1] = torch.as_tensor(
            [0.0 if done else 1.0 for done in batch_done], device=self.device
        )
        self.rewards[self.t - self.t_start - 1] = torch.as_tensor(
            batch_reward, device=self.device, dtype=torch.float
        )
        self.states[self.t - self.t_start] = statevar

        if self.t - self.t_start == self.update_steps:
            self.update()

    def get_statistics(self):
        return [
            ("average_actor", self.average_actor_loss),
            ("average_value", self.average_value),
            ("average_entropy", self.average_entropy),
        ]
