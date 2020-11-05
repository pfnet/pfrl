import copy
from logging import getLogger

import numpy as np
import torch
from torch import nn

from pfrl import agent
from pfrl.action_value import SingleActionValue
from pfrl.utils import clip_l2_grad_norm_, copy_param
from pfrl.utils.batch_states import batch_states
from pfrl.utils.mode_of_distribution import mode_of_distribution
from pfrl.utils.recurrent import detach_recurrent_state, one_step_forward


def compute_importance(pi, mu, x):
    with torch.no_grad():
        return float(torch.exp(pi.log_prob(x) - mu.log_prob(x)))


def compute_full_importance(pi, mu):
    assert isinstance(pi, torch.distributions.Categorical)
    assert isinstance(mu, torch.distributions.Categorical)
    # Categorical.logits is already normalized, i.e., exp(logits[i]) = probs[i]
    with torch.no_grad():
        pimu = torch.exp(pi.logits - mu.logits)
    return pimu


def compute_policy_gradient_full_correction(
    action_distrib, action_distrib_mu, action_value, v, truncation_threshold
):
    """Compute off-policy bias correction term wrt all actions."""
    assert isinstance(action_distrib, torch.distributions.Categorical)
    assert isinstance(action_distrib_mu, torch.distributions.Categorical)
    assert truncation_threshold is not None
    assert np.isscalar(v)
    with torch.no_grad():
        rho_all_inv = compute_full_importance(action_distrib_mu, action_distrib)
        correction_weight = (
            torch.nn.functional.relu(1 - truncation_threshold * rho_all_inv)
            * action_distrib.probs[0]
        )
        correction_advantage = action_value.q_values[0] - v
    # Categorical.logits is already normalized, i.e., logits[i] = log(probs[i])
    return -(correction_weight * action_distrib.logits * correction_advantage).sum(1)


def compute_policy_gradient_sample_correction(
    action_distrib, action_distrib_mu, action_value, v, truncation_threshold
):
    """Compute off-policy bias correction term wrt a sampled action."""
    assert np.isscalar(v)
    assert truncation_threshold is not None
    with torch.no_grad():
        sample_action = action_distrib.sample()
        rho_dash_inv = compute_importance(
            action_distrib_mu, action_distrib, sample_action
        )
        if truncation_threshold > 0 and rho_dash_inv >= 1 / truncation_threshold:
            return torch.as_tensor(0, dtype=torch.float)
        correction_weight = max(0, 1 - truncation_threshold * rho_dash_inv)
        assert correction_weight <= 1
        q = float(action_value.evaluate_actions(sample_action))
        correction_advantage = q - v
    return -(
        correction_weight
        * action_distrib.log_prob(sample_action)
        * correction_advantage
    )


def compute_policy_gradient_loss(
    action,
    advantage,
    action_distrib,
    action_distrib_mu,
    action_value,
    v,
    truncation_threshold,
):
    """Compute policy gradient loss with off-policy bias correction."""
    assert np.isscalar(advantage)
    assert np.isscalar(v)
    log_prob = action_distrib.log_prob(action)
    if action_distrib_mu is not None:
        # Off-policy
        rho = compute_importance(action_distrib, action_distrib_mu, action)
        g_loss = 0
        if truncation_threshold is None:
            g_loss -= rho * log_prob * advantage
        else:
            # Truncated off-policy policy gradient term
            g_loss -= min(truncation_threshold, rho) * log_prob * advantage
            # Bias correction term
            if isinstance(action_distrib, torch.distributions.Categorical):
                g_loss += compute_policy_gradient_full_correction(
                    action_distrib=action_distrib,
                    action_distrib_mu=action_distrib_mu,
                    action_value=action_value,
                    v=v,
                    truncation_threshold=truncation_threshold,
                )
            else:
                g_loss += compute_policy_gradient_sample_correction(
                    action_distrib=action_distrib,
                    action_distrib_mu=action_distrib_mu,
                    action_value=action_value,
                    v=v,
                    truncation_threshold=truncation_threshold,
                )
    else:
        # On-policy
        g_loss = -log_prob * advantage
    return g_loss


class ACERDiscreteActionHead(nn.Module):
    """ACER model that consists of a separate policy and V-function.

    Args:
        pi (Policy): Policy.
        q (QFunction): Q-function.
    """

    def __init__(self, pi, q):
        super().__init__()
        self.pi = pi
        self.q = q

    def forward(self, obs):
        action_distrib = self.pi(obs)
        action_value = self.q(obs)
        v = (action_distrib.probs * action_value.q_values).sum(1)
        return action_distrib, action_value, v


class ACERContinuousActionHead(nn.Module):
    """ACER model that consists of a separate policy and V-function.

    Args:
        pi (Policy): Policy.
        v (torch.nn.Module): V-function, a callable mapping from a batch of
            observations to a (batch_size, 1)-shaped `torch.Tensor`.
        adv (StateActionQFunction): Advantage function.
        n (int): Number of samples used to evaluate Q-values.
    """

    def __init__(self, pi, v, adv, n=5):
        super().__init__()
        self.pi = pi
        self.v = v
        self.adv = adv
        self.n = n

    def forward(self, obs):
        action_distrib = self.pi(obs)
        v = self.v(obs)

        def evaluator(action):
            adv_mean = (
                sum(self.adv((obs, action_distrib.sample())) for _ in range(self.n))
                / self.n
            )
            return v + self.adv((obs, action)) - adv_mean

        action_value = SingleActionValue(evaluator)

        return action_distrib, action_value, v


def get_params_of_distribution(distrib):
    if isinstance(distrib, torch.distributions.Independent):
        return get_params_of_distribution(distrib.base_dist)
    elif isinstance(distrib, torch.distributions.Categorical):
        return (distrib._param,)
    elif isinstance(distrib, torch.distributions.Normal):
        return distrib.loc, distrib.scale
    else:
        raise NotImplementedError("{} is not supported by ACER".format(type(distrib)))


def deepcopy_distribution(distrib):
    """Deepcopy a PyTorch distribution.

    PyTorch distributions cannot be deepcopied as it is except its tensors are
    graph leaves.
    """
    if isinstance(distrib, torch.distributions.Independent):
        return torch.distributions.Independent(
            deepcopy_distribution(distrib.base_dist), distrib.reinterpreted_batch_ndims,
        )
    elif isinstance(distrib, torch.distributions.Categorical):
        return torch.distributions.Categorical(logits=distrib.logits.clone().detach(),)
    elif isinstance(distrib, torch.distributions.Normal):
        return torch.distributions.Normal(
            loc=distrib.loc.clone().detach(), scale=distrib.scale.clone().detach(),
        )
    else:
        raise NotImplementedError("{} is not supported by ACER".format(type(distrib)))


def compute_loss_with_kl_constraint(distrib, another_distrib, original_loss, delta):
    """Compute loss considering a KL constraint.

    Args:
        distrib (Distribution): Distribution to optimize
        another_distrib (Distribution): Distribution used to compute KL
        original_loss (torch.Tensor): Loss to minimize
        delta (float): Minimum KL difference
    Returns:
        torch.Tensor: new loss to minimize
    """
    distrib_params = get_params_of_distribution(distrib)
    for param in distrib_params:
        assert param.shape[0] == 1
        assert param.requires_grad
    # Compute g: a direction to minimize the original loss
    g = [
        grad[0]
        for grad in torch.autograd.grad(
            [original_loss], distrib_params, retain_graph=True
        )
    ]

    # Compute k: a direction to increase KL div.
    kl = torch.distributions.kl_divergence(another_distrib, distrib)
    k = [
        grad[0]
        for grad in torch.autograd.grad([-kl], distrib_params, retain_graph=True)
    ]

    # Compute z: combination of g and k to keep small KL div.
    kg_dot = sum(torch.dot(kp.flatten(), gp.flatten()) for kp, gp in zip(k, g))
    kk_dot = sum(torch.dot(kp.flatten(), kp.flatten()) for kp in k)
    if kk_dot > 0:
        k_factor = max(0, ((kg_dot - delta) / kk_dot))
    else:
        k_factor = 0
    z = [gp - k_factor * kp for kp, gp in zip(k, g)]
    loss = 0
    for p, zp in zip(distrib_params, z):
        loss += (p * zp).sum()
    return loss.reshape(original_loss.shape), float(kl)


class ACER(agent.AttributeSavingMixin, agent.AsyncAgent):
    """ACER (Actor-Critic with Experience Replay).

    See http://arxiv.org/abs/1611.01224

    Args:
        model (ACERModel): Model to train. It must be a callable that accepts
            observations as input and return three values: action distributions
            (Distribution), Q values (ActionValue) and state values
            (torch.Tensor).
        optimizer (torch.optim.Optimizer): optimizer used to train the model
        t_max (int): The model is updated after every t_max local steps
        gamma (float): Discount factor [0,1]
        replay_buffer (EpisodicReplayBuffer): Replay buffer to use. If set
            None, this agent won't use experience replay.
        beta (float): Weight coefficient for the entropy regularizaiton term.
        phi (callable): Feature extractor function
        pi_loss_coef (float): Weight coefficient for the loss of the policy
        Q_loss_coef (float): Weight coefficient for the loss of the value
            function
        use_trust_region (bool): If set true, use efficient TRPO.
        trust_region_alpha (float): Decay rate of the average model used for
            efficient TRPO.
        trust_region_delta (float): Threshold used for efficient TRPO.
        truncation_threshold (float or None): Threshold used to truncate larger
            importance weights. If set None, importance weights are not
            truncated.
        disable_online_update (bool): If set true, disable online on-policy
            update and rely only on experience replay.
        n_times_replay (int): Number of times experience replay is repeated per
            one time of online update.
        replay_start_size (int): Experience replay is disabled if the number of
            transitions in the replay buffer is lower than this value.
        normalize_loss_by_steps (bool): If set true, losses are normalized by
            the number of steps taken to accumulate the losses
        act_deterministically (bool): If set true, choose most probable actions
            in act method.
        max_grad_norm (float or None): Maximum L2 norm of the gradient used for
            gradient clipping. If set to None, the gradient is not clipped.
        recurrent (bool): If set to True, `model` is assumed to implement
            `pfrl.nn.StatelessRecurrent`.
        use_Q_opc (bool): If set true, use Q_opc, a Q-value estimate without
            importance sampling, is used to compute advantage values for policy
            gradients. The original paper recommend to use in case of
            continuous action.
        average_entropy_decay (float): Decay rate of average entropy. Used only
            to record statistics.
        average_value_decay (float): Decay rate of average value. Used only
            to record statistics.
        average_kl_decay (float): Decay rate of kl value. Used only to record
            statistics.
    """

    process_idx = None
    saved_attributes = ("model", "optimizer")

    def __init__(
        self,
        model,
        optimizer,
        t_max,
        gamma,
        replay_buffer,
        beta=1e-2,
        phi=lambda x: x,
        pi_loss_coef=1.0,
        Q_loss_coef=0.5,
        use_trust_region=True,
        trust_region_alpha=0.99,
        trust_region_delta=1,
        truncation_threshold=10,
        disable_online_update=False,
        n_times_replay=8,
        replay_start_size=10 ** 4,
        normalize_loss_by_steps=True,
        act_deterministically=False,
        max_grad_norm=None,
        recurrent=False,
        use_Q_opc=False,
        average_entropy_decay=0.999,
        average_value_decay=0.999,
        average_kl_decay=0.999,
        logger=None,
    ):

        # Globally shared model
        self.shared_model = model

        # Globally shared average model used to compute trust regions
        self.shared_average_model = copy.deepcopy(self.shared_model)

        # Thread specific model
        self.model = copy.deepcopy(self.shared_model)

        self.optimizer = optimizer

        self.replay_buffer = replay_buffer
        self.t_max = t_max
        self.gamma = gamma
        self.beta = beta
        self.phi = phi
        self.pi_loss_coef = pi_loss_coef
        self.Q_loss_coef = Q_loss_coef
        self.normalize_loss_by_steps = normalize_loss_by_steps
        self.act_deterministically = act_deterministically
        self.max_grad_norm = max_grad_norm
        self.recurrent = recurrent
        self.use_trust_region = use_trust_region
        self.trust_region_alpha = trust_region_alpha
        self.truncation_threshold = truncation_threshold
        self.trust_region_delta = trust_region_delta
        self.disable_online_update = disable_online_update
        self.n_times_replay = n_times_replay
        self.use_Q_opc = use_Q_opc
        self.replay_start_size = replay_start_size
        self.average_value_decay = average_value_decay
        self.average_entropy_decay = average_entropy_decay
        self.average_kl_decay = average_kl_decay
        self.logger = logger if logger else getLogger(__name__)

        self.device = torch.device("cpu")
        self.t = 0
        self.last_state = None
        self.last_action = None

        # Recurrent states of the model
        self.train_recurrent_states = None
        self.shared_recurrent_states = None
        self.test_recurrent_states = None

        # Stats
        self.average_value = 0
        self.average_entropy = 0
        self.average_kl = 0

        self.init_history_data_for_online_update()

    def init_history_data_for_online_update(self):
        self.past_actions = {}
        self.past_rewards = {}
        self.past_values = {}
        self.past_action_distrib = {}
        self.past_action_values = {}
        self.past_avg_action_distrib = {}
        self.t_start = self.t

    def sync_parameters(self):
        copy_param.copy_param(target_link=self.model, source_link=self.shared_model)
        copy_param.soft_copy_param(
            target_link=self.shared_average_model,
            source_link=self.model,
            tau=1 - self.trust_region_alpha,
        )

    def assert_shared_memory(self):
        # Shared model must have tensors in shared memory
        for k, v in self.shared_model.state_dict().items():
            assert v.is_shared(), "{} is not in shared memory".format(k)

        # Local model must not have tensors in shared memory
        for k, v in self.model.state_dict().items():
            assert not v.is_shared(), "{} is in shared memory".format(k)

        # Optimizer must have tensors in shared memory
        for param_state in self.optimizer.state_dict()["state"].values():
            for k, v in param_state.items():
                if isinstance(v, torch.Tensor):
                    assert v.is_shared(), "{} is not in shared memory".format(k)

    @property
    def shared_attributes(self):
        return ("shared_model", "shared_average_model", "optimizer")

    def compute_one_step_pi_loss(
        self,
        action,
        advantage,
        action_distrib,
        action_distrib_mu,
        action_value,
        v,
        avg_action_distrib,
    ):
        assert np.isscalar(advantage)
        assert np.isscalar(v)

        g_loss = compute_policy_gradient_loss(
            action=action,
            advantage=advantage,
            action_distrib=action_distrib,
            action_distrib_mu=action_distrib_mu,
            action_value=action_value,
            v=v,
            truncation_threshold=self.truncation_threshold,
        )

        if self.use_trust_region:
            pi_loss, kl = compute_loss_with_kl_constraint(
                action_distrib,
                avg_action_distrib,
                g_loss,
                delta=self.trust_region_delta,
            )
            self.average_kl += (1 - self.average_kl_decay) * (kl - self.average_kl)
        else:
            pi_loss = g_loss

        # Entropy is maximized
        pi_loss -= self.beta * action_distrib.entropy()
        return pi_loss

    def compute_loss(
        self,
        t_start,
        t_stop,
        R,
        actions,
        rewards,
        values,
        action_values,
        action_distribs,
        action_distribs_mu,
        avg_action_distribs,
    ):

        assert np.isscalar(R)
        pi_loss = 0
        Q_loss = 0
        Q_ret = R
        Q_opc = R
        discrete = isinstance(action_distribs[t_start], torch.distributions.Categorical)
        del R
        for i in reversed(range(t_start, t_stop)):
            r = rewards[i]
            v = values[i]
            action_distrib = action_distribs[i]
            action_distrib_mu = action_distribs_mu[i] if action_distribs_mu else None
            avg_action_distrib = avg_action_distribs[i]
            action_value = action_values[i]
            ba = torch.as_tensor(actions[i]).unsqueeze(0)
            if action_distrib_mu is not None:
                # Off-policy
                rho = compute_importance(action_distrib, action_distrib_mu, ba)
            else:
                # On-policy
                rho = 1

            Q_ret = r + self.gamma * Q_ret
            Q_opc = r + self.gamma * Q_opc

            assert np.isscalar(Q_ret)
            assert np.isscalar(Q_opc)
            if self.use_Q_opc:
                advantage = Q_opc - float(v)
            else:
                advantage = Q_ret - float(v)
            pi_loss += self.compute_one_step_pi_loss(
                action=ba,
                advantage=advantage,
                action_distrib=action_distrib,
                action_distrib_mu=action_distrib_mu,
                action_value=action_value,
                v=float(v),
                avg_action_distrib=avg_action_distrib,
            )

            # Accumulate gradients of value function
            Q = action_value.evaluate_actions(ba)
            assert Q.requires_grad, "Q must be backprop-able"
            Q_loss += nn.functional.mse_loss(torch.tensor(Q_ret), Q) / 2

            if not discrete:
                assert v.requires_grad, "v must be backprop-able"
                v_target = min(1, rho) * (Q_ret - float(Q)) + float(v)
                Q_loss += nn.functional.mse_loss(torch.tensor(v_target), v) / 2

            if self.process_idx == 0:
                self.logger.debug(
                    "t:%s v:%s Q:%s Q_ret:%s Q_opc:%s",
                    i,
                    float(v),
                    float(Q),
                    Q_ret,
                    Q_opc,
                )

            if discrete:
                c = min(1, rho)
            else:
                c = min(1, rho ** (1 / ba.numel()))
            Q_ret = c * (Q_ret - float(Q)) + float(v)
            Q_opc = Q_opc - float(Q) + float(v)

        pi_loss *= self.pi_loss_coef
        Q_loss *= self.Q_loss_coef

        if self.normalize_loss_by_steps:
            pi_loss /= t_stop - t_start
            Q_loss /= t_stop - t_start

        if self.process_idx == 0:
            self.logger.debug("pi_loss:%s Q_loss:%s", float(pi_loss), float(Q_loss))

        return pi_loss + Q_loss.reshape(*pi_loss.shape)

    def update(
        self,
        t_start,
        t_stop,
        R,
        actions,
        rewards,
        values,
        action_values,
        action_distribs,
        action_distribs_mu,
        avg_action_distribs,
    ):

        assert np.isscalar(R)
        self.assert_shared_memory()

        total_loss = self.compute_loss(
            t_start=t_start,
            t_stop=t_stop,
            R=R,
            actions=actions,
            rewards=rewards,
            values=values,
            action_values=action_values,
            action_distribs=action_distribs,
            action_distribs_mu=action_distribs_mu,
            avg_action_distribs=avg_action_distribs,
        )

        # Compute gradients using thread-specific model
        self.model.zero_grad()
        total_loss.squeeze().backward()
        if self.max_grad_norm is not None:
            clip_l2_grad_norm_(self.model.parameters(), self.max_grad_norm)
        # Copy the gradients to the globally shared model
        copy_param.copy_grad(target_link=self.shared_model, source_link=self.model)
        self.optimizer.step()

        self.sync_parameters()

    def update_from_replay(self):

        if self.replay_buffer is None:
            return

        if len(self.replay_buffer) < self.replay_start_size:
            return

        episode = self.replay_buffer.sample_episodes(1, self.t_max)[0]

        model_recurrent_state = None
        shared_recurrent_state = None
        rewards = {}
        actions = {}
        action_distribs = {}
        action_distribs_mu = {}
        avg_action_distribs = {}
        action_values = {}
        values = {}
        for t, transition in enumerate(episode):
            bs = batch_states([transition["state"]], self.device, self.phi)
            if self.recurrent:
                (
                    (action_distrib, action_value, v),
                    model_recurrent_state,
                ) = one_step_forward(self.model, bs, model_recurrent_state)
            else:
                action_distrib, action_value, v = self.model(bs)
            with torch.no_grad():
                if self.recurrent:
                    (
                        (avg_action_distrib, _, _),
                        shared_recurrent_state,
                    ) = one_step_forward(
                        self.shared_average_model, bs, shared_recurrent_state,
                    )
                else:
                    avg_action_distrib, _, _ = self.shared_average_model(bs)
            actions[t] = transition["action"]
            values[t] = v
            action_distribs[t] = action_distrib
            avg_action_distribs[t] = avg_action_distrib
            rewards[t] = transition["reward"]
            action_distribs_mu[t] = transition["mu"]
            action_values[t] = action_value
        last_transition = episode[-1]
        if last_transition["is_state_terminal"]:
            R = 0
        else:
            with torch.no_grad():
                last_s = batch_states(
                    [last_transition["next_state"]], self.device, self.phi
                )
                if self.recurrent:
                    (_, _, last_v), _ = one_step_forward(
                        self.model, last_s, model_recurrent_state
                    )
                else:
                    _, _, last_v = self.model(last_s)
            R = float(last_v)
        return self.update(
            R=R,
            t_start=0,
            t_stop=len(episode),
            rewards=rewards,
            actions=actions,
            values=values,
            action_distribs=action_distribs,
            action_distribs_mu=action_distribs_mu,
            avg_action_distribs=avg_action_distribs,
            action_values=action_values,
        )

    def update_on_policy(self, statevar):
        assert self.t_start < self.t

        if not self.disable_online_update:
            if statevar is None:
                R = 0
            else:
                with torch.no_grad():
                    if self.recurrent:
                        (_, _, v), _ = one_step_forward(
                            self.model, statevar, self.train_recurrent_states
                        )
                    else:
                        _, _, v = self.model(statevar)
                R = float(v)
            self.update(
                t_start=self.t_start,
                t_stop=self.t,
                R=R,
                actions=self.past_actions,
                rewards=self.past_rewards,
                values=self.past_values,
                action_values=self.past_action_values,
                action_distribs=self.past_action_distrib,
                action_distribs_mu=None,
                avg_action_distribs=self.past_avg_action_distrib,
            )

        self.init_history_data_for_online_update()
        self.train_recurrent_states = detach_recurrent_state(
            self.train_recurrent_states
        )

    def act(self, obs):
        if self.training:
            return self._act_train(obs)
        else:
            return self._act_eval(obs)

    def observe(self, obs, reward, done, reset):
        if self.training:
            self._observe_train(obs, reward, done, reset)
        else:
            self._observe_eval(obs, reward, done, reset)

    def _act_train(self, obs):

        statevar = batch_states([obs], self.device, self.phi)

        if self.recurrent:
            (
                (action_distrib, action_value, v),
                self.train_recurrent_states,
            ) = one_step_forward(self.model, statevar, self.train_recurrent_states)
        else:
            action_distrib, action_value, v = self.model(statevar)
        self.past_action_values[self.t] = action_value
        action = action_distrib.sample()[0]

        # Save values for a later update
        self.past_values[self.t] = v
        self.past_action_distrib[self.t] = action_distrib
        with torch.no_grad():
            if self.recurrent:
                (
                    (avg_action_distrib, _, _),
                    self.shared_recurrent_states,
                ) = one_step_forward(
                    self.shared_average_model, statevar, self.shared_recurrent_states,
                )
            else:
                avg_action_distrib, _, _ = self.shared_average_model(statevar)
        self.past_avg_action_distrib[self.t] = avg_action_distrib

        self.past_actions[self.t] = action

        # Update stats
        self.average_value += (1 - self.average_value_decay) * (
            float(v) - self.average_value
        )
        self.average_entropy += (1 - self.average_entropy_decay) * (
            float(action_distrib.entropy()) - self.average_entropy
        )

        self.last_state = obs
        self.last_action = action.numpy()
        self.last_action_distrib = deepcopy_distribution(action_distrib)

        return self.last_action

    def _act_eval(self, obs):
        # Use the process-local model for acting
        with torch.no_grad():
            statevar = batch_states([obs], self.device, self.phi)
            if self.recurrent:
                (action_distrib, _, _), self.test_recurrent_states = one_step_forward(
                    self.model, statevar, self.test_recurrent_states
                )
            else:
                action_distrib, _, _ = self.model(statevar)
            if self.act_deterministically:
                return mode_of_distribution(action_distrib).numpy()[0]
            else:
                return action_distrib.sample().numpy()[0]

    def _observe_train(self, state, reward, done, reset):
        assert self.last_state is not None
        assert self.last_action is not None

        # Add a transition to the replay buffer
        if self.replay_buffer is not None:
            self.replay_buffer.append(
                state=self.last_state,
                action=self.last_action,
                reward=reward,
                next_state=state,
                is_state_terminal=done,
                mu=self.last_action_distrib,
            )
            if done or reset:
                self.replay_buffer.stop_current_episode()

        self.t += 1
        self.past_rewards[self.t - 1] = reward
        if self.process_idx == 0:
            self.logger.debug(
                "t:%s r:%s a:%s", self.t, reward, self.last_action,
            )

        if self.t - self.t_start == self.t_max or done or reset:
            if done:
                statevar = None
            else:
                statevar = batch_states([state], self.device, self.phi)
            self.update_on_policy(statevar)
            for _ in range(self.n_times_replay):
                self.update_from_replay()
        if done or reset:
            self.train_recurrent_states = None
            self.shared_recurrent_states = None

        self.last_state = None
        self.last_action = None
        self.last_action_distrib = None

    def _observe_eval(self, obs, reward, done, reset):
        if done or reset:
            self.test_recurrent_states = None

    def load(self, dirname):
        super().load(dirname)
        copy_param.copy_param(target_link=self.shared_model, source_link=self.model)

    def get_statistics(self):
        return [
            ("average_value", self.average_value),
            ("average_entropy", self.average_entropy),
            ("average_kl", self.average_kl),
        ]
