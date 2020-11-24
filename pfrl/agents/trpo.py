import collections
import random
from logging import getLogger

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import pfrl
from pfrl import agent
from pfrl.agents.ppo import (  # NOQA
    _compute_explained_variance,
    _make_dataset,
    _make_dataset_recurrent,
    _yield_minibatches,
    _yield_subset_of_sequences_with_fixed_number_of_items,
)
from pfrl.utils import clip_l2_grad_norm_
from pfrl.utils.batch_states import batch_states
from pfrl.utils.mode_of_distribution import mode_of_distribution
from pfrl.utils.recurrent import (
    concatenate_recurrent_states,
    flatten_sequences_time_first,
    get_recurrent_state_at,
    mask_recurrent_state_at,
    one_step_forward,
    pack_and_forward,
)


def _flatten_and_concat_variables(vs):
    """Flatten and concat variables to make a single flat vector variable."""
    return torch.cat([torch.flatten(v) for v in vs], dim=0)


def _as_ndarray(x):
    """torch.Tensor or ndarray -> ndarray."""
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    else:
        return x


def _split_and_reshape_to_ndarrays(flat_v, sizes, shapes):
    """Split and reshape a single flat vector to make a list of ndarrays."""
    vs = torch.split(flat_v, sizes)
    return [v.reshape(shape) for v, shape in zip(vs, shapes)]


def _replace_params_data(params, new_params_data):
    """Replace data of params with new data."""
    for param, new_param_data in zip(params, new_params_data):
        assert param.shape == new_param_data.shape
        assert isinstance(param, nn.Parameter)
        param.data.copy_(new_param_data)


def _hessian_vector_product(flat_grads, params, vec):
    """Compute hessian vector product efficiently by backprop."""
    vec = vec.detach()
    grads = torch.autograd.grad(
        [torch.sum(flat_grads * vec)], params, retain_graph=True
    )
    assert all(
        grad is not None for grad in grads
    ), "The Hessian-vector product contains None."
    return _flatten_and_concat_variables(grads)


def _mean_or_nan(xs):
    """Return its mean a non-empty sequence, numpy.nan for a empty one."""
    return np.mean(xs) if xs else np.nan


def _collect_first_recurrent_states_of_policy(episodes):
    return [
        (ep[0]["recurrent_state"][0] if ep[0]["recurrent_state"] is not None else None)
        for ep in episodes
    ]


def _collect_first_recurrent_states_of_vf(episodes):
    return [
        (ep[0]["recurrent_state"][1] if ep[0]["recurrent_state"] is not None else None)
        for ep in episodes
    ]


class TRPO(agent.AttributeSavingMixin, agent.BatchAgent):
    """Trust Region Policy Optimization.

    A given stochastic policy is optimized by the TRPO algorithm. A given
    value function is also trained to predict by the TD(lambda) algorithm and
    used for Generalized Advantage Estimation (GAE).

    Since the policy is optimized via the conjugate gradient method and line
    search while the value function is optimized via SGD, these two models
    should be separate.

    Since TRPO requires second-order derivatives to compute Hessian-vector
    products, your policy must contain only functions that support second-order
    derivatives.

    See https://arxiv.org/abs/1502.05477 for TRPO.
    See https://arxiv.org/abs/1506.02438 for GAE.

    Args:
        policy (Policy): Stochastic policy. Its forward computation must
            contain only functions that support second-order derivatives.
            Recurrent models are not supported.
        vf (ValueFunction): Value function. Recurrent models are not supported.
        vf_optimizer (torch.optim.Optimizer): Optimizer for the value function.
        obs_normalizer (pfrl.nn.EmpiricalNormalization or None):
            If set to pfrl.nn.EmpiricalNormalization, it is used to
            normalize observations based on the empirical mean and standard
            deviation of observations. These statistics are updated after
            computing advantages and target values and before updating the
            policy and the value function.
        gpu (int): GPU device id if not None nor negative
        gamma (float): Discount factor [0, 1]
        lambd (float): Lambda-return factor [0, 1]
        phi (callable): Feature extractor function
        entropy_coef (float): Weight coefficient for entropy bonus [0, inf)
        update_interval (int): Interval steps of TRPO iterations. Every time after
            this amount of steps, this agent updates the policy and the value
            function using data from these steps.
        vf_epochs (int): Number of epochs for which the value function is
            trained on each TRPO iteration.
        vf_batch_size (int): Batch size of SGD for the value function.
        standardize_advantages (bool): Use standardized advantages on updates
        line_search_max_backtrack (int): Maximum number of backtracking in line
            search to tune step sizes of policy updates.
        conjugate_gradient_max_iter (int): Maximum number of iterations in
            the conjugate gradient method.
        conjugate_gradient_damping (float): Damping factor used in the
            conjugate gradient method.
        act_deterministically (bool): If set to True, choose most probable
            actions in the act method instead of sampling from distributions.
        max_grad_norm (float or None): Maximum L2 norm of the gradient used for
            gradient clipping. If set to None, the gradient is not clipped.
        value_stats_window (int): Window size used to compute statistics
            of value predictions.
        entropy_stats_window (int): Window size used to compute statistics
            of entropy of action distributions.
        kl_stats_window (int): Window size used to compute statistics
            of KL divergence between old and new policies.
        policy_step_size_stats_window (int): Window size used to compute
            statistics of step sizes of policy updates.

    Statistics:
        average_value: Average of value predictions on non-terminal states.
            It's updated after `act` or `batch_act` methods are called in the
            training mode.
        average_entropy: Average of entropy of action distributions on
            non-terminal states. It's updated after `act` or `batch_act`
            methods are called in the training mode.
        average_kl: Average of KL divergence between old and new policies.
            It's updated after the policy is updated.
        average_policy_step_size: Average of step sizes of policy updates
            It's updated after the policy is updated.
    """

    saved_attributes = ("policy", "vf", "vf_optimizer", "obs_normalizer")

    def __init__(
        self,
        policy,
        vf,
        vf_optimizer,
        obs_normalizer=None,
        gpu=None,
        gamma=0.99,
        lambd=0.95,
        phi=lambda x: x,
        entropy_coef=0.01,
        update_interval=2048,
        max_kl=0.01,
        vf_epochs=3,
        vf_batch_size=64,
        standardize_advantages=True,
        batch_states=batch_states,
        recurrent=False,
        max_recurrent_sequence_len=None,
        line_search_max_backtrack=10,
        conjugate_gradient_max_iter=10,
        conjugate_gradient_damping=1e-2,
        act_deterministically=False,
        max_grad_norm=None,
        value_stats_window=1000,
        entropy_stats_window=1000,
        kl_stats_window=100,
        policy_step_size_stats_window=100,
        logger=getLogger(__name__),
    ):

        self.policy = policy
        self.vf = vf
        self.vf_optimizer = vf_optimizer
        self.obs_normalizer = obs_normalizer
        self.gamma = gamma
        self.lambd = lambd
        self.phi = phi
        self.entropy_coef = entropy_coef
        self.update_interval = update_interval
        self.max_kl = max_kl
        self.vf_epochs = vf_epochs
        self.vf_batch_size = vf_batch_size
        self.standardize_advantages = standardize_advantages
        self.batch_states = batch_states
        self.recurrent = recurrent
        self.max_recurrent_sequence_len = max_recurrent_sequence_len
        self.line_search_max_backtrack = line_search_max_backtrack
        self.conjugate_gradient_max_iter = conjugate_gradient_max_iter
        self.conjugate_gradient_damping = conjugate_gradient_damping
        self.act_deterministically = act_deterministically
        self.max_grad_norm = max_grad_norm
        self.logger = logger

        if gpu is not None and gpu >= 0:
            assert torch.cuda.is_available()
            self.device = torch.device("cuda:{}".format(gpu))
            self.policy.to(self.device)
            self.vf.to(self.device)
            if self.obs_normalizer is not None:
                self.obs_normalizer.to(self.device)
        else:
            self.device = torch.device("cpu")

        if recurrent:
            self.model = pfrl.nn.RecurrentBranched(policy, vf)
        else:
            self.model = pfrl.nn.Branched(policy, vf)

        self.value_record = collections.deque(maxlen=value_stats_window)
        self.entropy_record = collections.deque(maxlen=entropy_stats_window)
        self.kl_record = collections.deque(maxlen=kl_stats_window)
        self.policy_step_size_record = collections.deque(
            maxlen=policy_step_size_stats_window
        )
        self.explained_variance = np.nan

        self.last_state = None
        self.last_action = None

        # Contains episodes used for next update iteration
        self.memory = []
        # Contains transitions of the last episode not moved to self.memory yet
        self.last_episode = []

        # Batch versions of last_episode, last_state, and last_action
        self.batch_last_episode = None
        self.batch_last_state = None
        self.batch_last_action = None

        # Recurrent states of the model
        self.train_recurrent_states = None
        self.train_prev_recurrent_states = None
        self.test_recurrent_states = None

    def _initialize_batch_variables(self, num_envs):
        self.batch_last_episode = [[] for _ in range(num_envs)]
        self.batch_last_state = [None] * num_envs
        self.batch_last_action = [None] * num_envs

    def _update_if_dataset_is_ready(self):
        dataset_size = (
            sum(len(episode) for episode in self.memory)
            + len(self.last_episode)
            + (
                0
                if self.batch_last_episode is None
                else sum(len(episode) for episode in self.batch_last_episode)
            )
        )
        if dataset_size >= self.update_interval:
            self._flush_last_episode()
            if self.recurrent:
                dataset = _make_dataset_recurrent(
                    episodes=self.memory,
                    model=self.model,
                    phi=self.phi,
                    batch_states=self.batch_states,
                    obs_normalizer=self.obs_normalizer,
                    gamma=self.gamma,
                    lambd=self.lambd,
                    max_recurrent_sequence_len=self.max_recurrent_sequence_len,
                    device=self.device,
                )
                self._update_recurrent(dataset)
            else:
                dataset = _make_dataset(
                    episodes=self.memory,
                    model=self.model,
                    phi=self.phi,
                    batch_states=self.batch_states,
                    obs_normalizer=self.obs_normalizer,
                    gamma=self.gamma,
                    lambd=self.lambd,
                    device=self.device,
                )
                assert len(dataset) == dataset_size
                self._update(dataset)
            self.explained_variance = _compute_explained_variance(
                flatten_sequences_time_first(self.memory)
            )
            self.memory = []

    def _flush_last_episode(self):
        if self.last_episode:
            self.memory.append(self.last_episode)
            self.last_episode = []
        if self.batch_last_episode:
            for i, episode in enumerate(self.batch_last_episode):
                if episode:
                    self.memory.append(episode)
                    self.batch_last_episode[i] = []

    def _update(self, dataset):
        """Update both the policy and the value function."""

        if self.obs_normalizer:
            self._update_obs_normalizer(dataset)
        self._update_policy(dataset)
        self._update_vf(dataset)

    def _update_recurrent(self, dataset):
        """Update both the policy and the value function."""

        flat_dataset = flatten_sequences_time_first(dataset)
        if self.obs_normalizer:
            self._update_obs_normalizer(flat_dataset)

        self._update_policy_recurrent(dataset)
        self._update_vf_recurrent(dataset)

    def _update_vf_recurrent(self, dataset):

        for epoch in range(self.vf_epochs):
            random.shuffle(dataset)
            for (
                minibatch
            ) in _yield_subset_of_sequences_with_fixed_number_of_items(  # NOQA
                dataset, self.vf_batch_size
            ):
                self._update_vf_once_recurrent(minibatch)

    def _update_vf_once_recurrent(self, episodes):

        # Sort episodes desc by length for pack_sequence
        episodes = sorted(episodes, key=len, reverse=True)

        flat_transitions = flatten_sequences_time_first(episodes)

        # Prepare data for a recurrent model
        seqs_states = []
        for ep in episodes:
            states = self.batch_states(
                [transition["state"] for transition in ep], self.device, self.phi
            )
            if self.obs_normalizer:
                states = self.obs_normalizer(states, update=False)
            seqs_states.append(states)

        flat_vs_teacher = torch.as_tensor(
            [[transition["v_teacher"]] for transition in flat_transitions],
            device=self.device,
            dtype=torch.float,
        )

        with torch.no_grad():
            vf_rs = concatenate_recurrent_states(
                _collect_first_recurrent_states_of_vf(episodes)
            )

        flat_vs_pred, _ = pack_and_forward(self.vf, seqs_states, vf_rs)

        vf_loss = F.mse_loss(flat_vs_pred, flat_vs_teacher)
        self.vf.zero_grad()
        vf_loss.backward()
        if self.max_grad_norm is not None:
            clip_l2_grad_norm_(self.vf.parameters(), self.max_grad_norm)
        self.vf_optimizer.step()

    def _update_obs_normalizer(self, dataset):
        assert self.obs_normalizer
        states = batch_states([b["state"] for b in dataset], self.device, self.phi)
        self.obs_normalizer.experience(states)

    def _update_vf(self, dataset):
        """Update the value function using a given dataset.

        The value function is updated via SGD to minimize TD(lambda) errors.
        """

        assert "state" in dataset[0]
        assert "v_teacher" in dataset[0]

        for batch in _yield_minibatches(
            dataset, minibatch_size=self.vf_batch_size, num_epochs=self.vf_epochs
        ):
            states = batch_states([b["state"] for b in batch], self.device, self.phi)
            if self.obs_normalizer:
                states = self.obs_normalizer(states, update=False)
            vs_teacher = torch.as_tensor(
                [b["v_teacher"] for b in batch], device=self.device, dtype=torch.float,
            )
            vs_pred = self.vf(states)
            vf_loss = F.mse_loss(vs_pred, vs_teacher[..., None])
            self.vf.zero_grad()
            vf_loss.backward()
            if self.max_grad_norm is not None:
                clip_l2_grad_norm_(self.vf.parameters(), self.max_grad_norm)
            self.vf_optimizer.step()

    def _compute_gain(self, log_prob, log_prob_old, entropy, advs):
        """Compute a gain to maximize."""
        prob_ratio = torch.exp(log_prob - log_prob_old)
        mean_entropy = torch.mean(entropy)
        surrogate_gain = torch.mean(prob_ratio * advs)
        return surrogate_gain + self.entropy_coef * mean_entropy

    def _update_policy(self, dataset):
        """Update the policy using a given dataset.

        The policy is updated via CG and line search.
        """

        assert "state" in dataset[0]
        assert "action" in dataset[0]
        assert "adv" in dataset[0]

        # Use full-batch
        states = batch_states([b["state"] for b in dataset], self.device, self.phi)
        if self.obs_normalizer:
            states = self.obs_normalizer(states, update=False)
        actions = torch.as_tensor([b["action"] for b in dataset], device=self.device)
        advs = torch.as_tensor(
            [b["adv"] for b in dataset], device=self.device, dtype=torch.float
        )
        if self.standardize_advantages:
            std_advs, mean_advs = torch.std_mean(advs, unbiased=False)
            advs = (advs - mean_advs) / (std_advs + 1e-8)

        # Recompute action distributions for batch backprop
        action_distrib = self.policy(states)

        log_prob_old = torch.as_tensor(
            [transition["log_prob"] for transition in dataset],
            device=self.device,
            dtype=torch.float,
        )

        gain = self._compute_gain(
            log_prob=action_distrib.log_prob(actions),
            log_prob_old=log_prob_old,
            entropy=action_distrib.entropy(),
            advs=advs,
        )

        # Distribution to compute KL div against
        with torch.no_grad():
            # torch.distributions.Distribution cannot be deepcopied
            action_distrib_old = self.policy(states)

        full_step = self._compute_kl_constrained_step(
            action_distrib=action_distrib,
            action_distrib_old=action_distrib_old,
            gain=gain,
        )

        self._line_search(
            full_step=full_step,
            dataset=dataset,
            advs=advs,
            action_distrib_old=action_distrib_old,
            gain=gain,
        )

    def _update_policy_recurrent(self, dataset):
        """Update the policy using a given dataset.

        The policy is updated via CG and line search.
        """

        # Sort episodes desc by length for pack_sequence
        dataset = sorted(dataset, key=len, reverse=True)

        flat_transitions = flatten_sequences_time_first(dataset)

        # Prepare data for a recurrent model
        seqs_states = []
        for ep in dataset:
            states = self.batch_states(
                [transition["state"] for transition in ep], self.device, self.phi,
            )
            if self.obs_normalizer:
                states = self.obs_normalizer(states, update=False)
            seqs_states.append(states)

        flat_actions = torch.as_tensor(
            [transition["action"] for transition in flat_transitions],
            device=self.device,
        )
        flat_advs = torch.as_tensor(
            [transition["adv"] for transition in flat_transitions],
            device=self.device,
            dtype=torch.float,
        )

        if self.standardize_advantages:
            std_advs, mean_advs = torch.std_mean(flat_advs, unbiased=False)
            flat_advs = (flat_advs - mean_advs) / (std_advs + 1e-8)

        with torch.no_grad():
            policy_rs = concatenate_recurrent_states(
                _collect_first_recurrent_states_of_policy(dataset)
            )

        flat_distribs, _ = pack_and_forward(self.policy, seqs_states, policy_rs)

        log_prob_old = torch.tensor(
            [transition["log_prob"] for transition in flat_transitions],
            device=self.device,
            dtype=torch.float,
        )

        gain = self._compute_gain(
            log_prob=flat_distribs.log_prob(flat_actions),
            log_prob_old=log_prob_old,
            entropy=flat_distribs.entropy(),
            advs=flat_advs,
        )

        # Distribution to compute KL div against
        with torch.no_grad():
            # torch.distributions.Distribution cannot be deepcopied
            action_distrib_old, _ = pack_and_forward(
                self.policy, seqs_states, policy_rs
            )

        full_step = self._compute_kl_constrained_step(
            action_distrib=flat_distribs,
            action_distrib_old=action_distrib_old,
            gain=gain,
        )

        self._line_search(
            full_step=full_step,
            dataset=dataset,
            advs=flat_advs,
            action_distrib_old=action_distrib_old,
            gain=gain,
        )

    def _compute_kl_constrained_step(self, action_distrib, action_distrib_old, gain):
        """Compute a step of policy parameters with a KL constraint."""
        policy_params = list(self.policy.parameters())
        kl = torch.mean(
            torch.distributions.kl_divergence(action_distrib_old, action_distrib)
        )

        kl_grads = torch.autograd.grad([kl], policy_params, create_graph=True)
        assert all(
            g is not None for g in kl_grads
        ), "The gradient contains None. The policy may have unused parameters."
        flat_kl_grads = _flatten_and_concat_variables(kl_grads)
        assert all(g.requires_grad for g in kl_grads)
        assert flat_kl_grads.requires_grad

        def fisher_vector_product_func(vec):
            vec = torch.as_tensor(vec)
            fvp = _hessian_vector_product(flat_kl_grads, policy_params, vec)
            return fvp + self.conjugate_gradient_damping * vec

        gain_grads = torch.autograd.grad([gain], policy_params, create_graph=True)
        assert all(
            g is not None for g in gain_grads
        ), "The gradient contains None. The policy may have unused parameters."
        flat_gain_grads = _flatten_and_concat_variables(gain_grads).detach()
        step_direction = pfrl.utils.conjugate_gradient(
            fisher_vector_product_func,
            flat_gain_grads,
            max_iter=self.conjugate_gradient_max_iter,
        )

        # We want a step size that satisfies KL(old|new) < max_kl.
        # Let d = alpha * step_direction be the actual parameter updates.
        # The second-order approximation of KL divergence is:
        #   KL(old|new) = 1/2 d^T I d + O(||d||^3),
        # where I is a Fisher information matrix.
        # Substitute d = alpha * step_direction and solve KL(old|new) = max_kl
        # for alpha to get the step size that tightly satisfies the constraint.

        dId = float(step_direction.dot(fisher_vector_product_func(step_direction)))
        scale = (2.0 * self.max_kl / (dId + 1e-8)) ** 0.5
        return scale * step_direction

    def _line_search(self, full_step, dataset, advs, action_distrib_old, gain):
        """Do line search for a safe step size."""
        policy_params = list(self.policy.parameters())
        policy_params_sizes = [param.numel() for param in policy_params]
        policy_params_shapes = [param.shape for param in policy_params]
        step_size = 1.0
        flat_params = _flatten_and_concat_variables(policy_params).detach()

        if self.recurrent:
            seqs_states = []
            for ep in dataset:
                states = self.batch_states(
                    [transition["state"] for transition in ep], self.device, self.phi
                )
                if self.obs_normalizer:
                    states = self.obs_normalizer(states, update=False)
                seqs_states.append(states)
            with torch.no_grad(), pfrl.utils.evaluating(self.model):
                policy_rs = concatenate_recurrent_states(
                    _collect_first_recurrent_states_of_policy(dataset)
                )

            def evaluate_current_policy():
                distrib, _ = pack_and_forward(self.policy, seqs_states, policy_rs)
                return distrib

        else:
            states = self.batch_states(
                [transition["state"] for transition in dataset], self.device, self.phi
            )
            if self.obs_normalizer:
                states = self.obs_normalizer(states, update=False)

            def evaluate_current_policy():
                return self.policy(states)

        flat_transitions = (
            flatten_sequences_time_first(dataset) if self.recurrent else dataset
        )
        actions = torch.tensor(
            [transition["action"] for transition in flat_transitions],
            device=self.device,
        )
        log_prob_old = torch.tensor(
            [transition["log_prob"] for transition in flat_transitions],
            device=self.device,
            dtype=torch.float,
        )

        for i in range(self.line_search_max_backtrack + 1):
            self.logger.info("Line search iteration: %s step size: %s", i, step_size)
            new_flat_params = flat_params + step_size * full_step
            new_params = _split_and_reshape_to_ndarrays(
                new_flat_params, sizes=policy_params_sizes, shapes=policy_params_shapes,
            )
            _replace_params_data(policy_params, new_params)
            with torch.no_grad(), pfrl.utils.evaluating(self.policy):
                new_action_distrib = evaluate_current_policy()
                new_gain = self._compute_gain(
                    log_prob=new_action_distrib.log_prob(actions),
                    log_prob_old=log_prob_old,
                    entropy=new_action_distrib.entropy(),
                    advs=advs,
                )
                new_kl = torch.mean(
                    torch.distributions.kl_divergence(
                        action_distrib_old, new_action_distrib
                    )
                )

            improve = float(new_gain) - float(gain)
            self.logger.info("Surrogate objective improve: %s", improve)
            self.logger.info("KL divergence: %s", float(new_kl))
            if not torch.isfinite(new_gain):
                self.logger.info("Surrogate objective is not finite. Bakctracking...")
            elif not torch.isfinite(new_kl):
                self.logger.info("KL divergence is not finite. Bakctracking...")
            elif improve < 0:
                self.logger.info("Surrogate objective didn't improve. Bakctracking...")
            elif float(new_kl) > self.max_kl:
                self.logger.info("KL divergence exceeds max_kl. Bakctracking...")
            else:
                self.kl_record.append(float(new_kl))
                self.policy_step_size_record.append(step_size)
                break
            step_size *= 0.5
        else:
            self.logger.info(
                "Line search coundn't find a good step size. The policy was not"
                " updated."
            )
            self.policy_step_size_record.append(0.0)
            _replace_params_data(
                policy_params,
                _split_and_reshape_to_ndarrays(
                    flat_params, sizes=policy_params_sizes, shapes=policy_params_shapes
                ),
            )

    def batch_act(self, batch_obs):
        if self.training:
            return self._batch_act_train(batch_obs)
        else:
            return self._batch_act_eval(batch_obs)

    def batch_observe(self, batch_obs, batch_reward, batch_done, batch_reset):
        if self.training:
            self._batch_observe_train(batch_obs, batch_reward, batch_done, batch_reset)
        else:
            self._batch_observe_eval(batch_obs, batch_reward, batch_done, batch_reset)

    def _batch_act_eval(self, batch_obs):
        assert not self.training
        b_state = self.batch_states(batch_obs, self.device, self.phi)

        if self.obs_normalizer:
            b_state = self.obs_normalizer(b_state, update=False)

        with torch.no_grad(), pfrl.utils.evaluating(self.model):
            if self.recurrent:
                (action_distrib, _), self.test_recurrent_states = one_step_forward(
                    self.model, b_state, self.test_recurrent_states
                )
            else:
                action_distrib, _ = self.model(b_state)
            if self.act_deterministically:
                action = mode_of_distribution(action_distrib).cpu().numpy()
            else:
                action = action_distrib.sample().cpu().numpy()

        return action

    def _batch_act_train(self, batch_obs):
        assert self.training
        b_state = self.batch_states(batch_obs, self.device, self.phi)

        if self.obs_normalizer:
            b_state = self.obs_normalizer(b_state, update=False)

        num_envs = len(batch_obs)
        if self.batch_last_episode is None:
            self._initialize_batch_variables(num_envs)
        assert len(self.batch_last_episode) == num_envs
        assert len(self.batch_last_state) == num_envs
        assert len(self.batch_last_action) == num_envs

        # action_distrib will be recomputed when computing gradients
        with torch.no_grad(), pfrl.utils.evaluating(self.model):
            if self.recurrent:
                assert self.train_prev_recurrent_states is None
                self.train_prev_recurrent_states = self.train_recurrent_states
                (
                    (action_distrib, batch_value),
                    self.train_recurrent_states,
                ) = one_step_forward(
                    self.model, b_state, self.train_prev_recurrent_states
                )
            else:
                action_distrib, batch_value = self.model(b_state)
            batch_action = action_distrib.sample().cpu().numpy()
            self.entropy_record.extend(action_distrib.entropy().cpu().numpy())
            self.value_record.extend(batch_value.cpu().numpy())

        self.batch_last_state = list(batch_obs)
        self.batch_last_action = list(batch_action)

        return batch_action

    def _batch_observe_eval(self, batch_obs, batch_reward, batch_done, batch_reset):
        assert not self.training
        if self.recurrent:
            # Reset recurrent states when episodes end
            indices_that_ended = [
                i
                for i, (done, reset) in enumerate(zip(batch_done, batch_reset))
                if done or reset
            ]
            if indices_that_ended:
                self.test_recurrent_states = mask_recurrent_state_at(
                    self.test_recurrent_states, indices_that_ended
                )

    def _batch_observe_train(self, batch_obs, batch_reward, batch_done, batch_reset):
        assert self.training

        for i, (state, action, reward, next_state, done, reset) in enumerate(
            zip(  # NOQA
                self.batch_last_state,
                self.batch_last_action,
                batch_reward,
                batch_obs,
                batch_done,
                batch_reset,
            )
        ):
            if state is not None:
                assert action is not None
                transition = {
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "nonterminal": 0.0 if done else 1.0,
                }
                if self.recurrent:
                    transition["recurrent_state"] = get_recurrent_state_at(
                        self.train_prev_recurrent_states, i, detach=True
                    )
                    transition["next_recurrent_state"] = get_recurrent_state_at(
                        self.train_recurrent_states, i, detach=True
                    )
                self.batch_last_episode[i].append(transition)
            if done or reset:
                assert self.batch_last_episode[i]
                self.memory.append(self.batch_last_episode[i])
                self.batch_last_episode[i] = []
            self.batch_last_state[i] = None
            self.batch_last_action[i] = None

        self.train_prev_recurrent_states = None

        if self.recurrent:
            # Reset recurrent states when episodes end
            indices_that_ended = [
                i
                for i, (done, reset) in enumerate(zip(batch_done, batch_reset))
                if done or reset
            ]
            if indices_that_ended:
                self.train_recurrent_states = mask_recurrent_state_at(
                    self.train_recurrent_states, indices_that_ended
                )

        self._update_if_dataset_is_ready()

    def get_statistics(self):
        return [
            ("average_value", _mean_or_nan(self.value_record)),
            ("average_entropy", _mean_or_nan(self.entropy_record)),
            ("average_kl", _mean_or_nan(self.kl_record)),
            ("average_policy_step_size", _mean_or_nan(self.policy_step_size_record)),
            ("explained_variance", self.explained_variance),
        ]
