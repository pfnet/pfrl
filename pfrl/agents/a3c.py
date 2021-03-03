import copy
from logging import getLogger

import torch
import torch.nn.functional as F

import pfrl
from pfrl import agent
from pfrl.utils import clip_l2_grad_norm_, copy_param
from pfrl.utils.batch_states import batch_states
from pfrl.utils.mode_of_distribution import mode_of_distribution
from pfrl.utils.recurrent import one_step_forward, pack_and_forward

logger = getLogger(__name__)


class A3C(agent.AttributeSavingMixin, agent.AsyncAgent):
    """A3C: Asynchronous Advantage Actor-Critic.

    See http://arxiv.org/abs/1602.01783

    Args:
        model (A3CModel): Model to train
        optimizer (torch.optim.Optimizer): optimizer used to train the model
        t_max (int): The model is updated after every t_max local steps
        gamma (float): Discount factor [0,1]
        beta (float): Weight coefficient for the entropy regularizaiton term.
        process_idx (int): Index of the process.
        phi (callable): Feature extractor function
        pi_loss_coef (float): Weight coefficient for the loss of the policy
        v_loss_coef (float): Weight coefficient for the loss of the value
            function
        act_deterministically (bool): If set true, choose most probable actions
            in act method.
        max_grad_norm (float or None): Maximum L2 norm of the gradient used for
            gradient clipping. If set to None, the gradient is not clipped.
        recurrent (bool): If set to True, `model` is assumed to implement
            `pfrl.nn.StatelessRecurrent`.
        batch_states (callable): method which makes a batch of observations.
            default is `pfrl.utils.batch_states.batch_states`
    """

    process_idx = None
    saved_attributes = ("model", "optimizer")

    def __init__(
        self,
        model,
        optimizer,
        t_max,
        gamma,
        beta=1e-2,
        process_idx=0,
        phi=lambda x: x,
        pi_loss_coef=1.0,
        v_loss_coef=0.5,
        keep_loss_scale_same=False,
        normalize_grad_by_t_max=False,
        use_average_reward=False,
        act_deterministically=False,
        max_grad_norm=None,
        recurrent=False,
        average_entropy_decay=0.999,
        average_value_decay=0.999,
        batch_states=batch_states,
    ):

        # Globally shared model
        self.shared_model = model

        # Thread specific model
        self.model = copy.deepcopy(self.shared_model)

        self.optimizer = optimizer

        self.t_max = t_max
        self.gamma = gamma
        self.beta = beta
        self.phi = phi
        self.pi_loss_coef = pi_loss_coef
        self.v_loss_coef = v_loss_coef
        self.keep_loss_scale_same = keep_loss_scale_same
        self.normalize_grad_by_t_max = normalize_grad_by_t_max
        self.use_average_reward = use_average_reward
        self.act_deterministically = act_deterministically
        self.max_grad_norm = max_grad_norm
        self.recurrent = recurrent
        self.average_value_decay = average_value_decay
        self.average_entropy_decay = average_entropy_decay
        self.batch_states = batch_states

        self.device = torch.device("cpu")
        self.t = 0
        self.t_start = 0
        self.past_obs = {}
        self.past_action = {}
        self.past_rewards = {}
        self.past_recurrent_state = {}
        self.average_reward = 0

        # Recurrent states of the model
        self.train_recurrent_states = None
        self.test_recurrent_states = None

        # Stats
        self.average_value = 0
        self.average_entropy = 0

    def sync_parameters(self):
        copy_param.copy_param(target_link=self.model, source_link=self.shared_model)

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
        return ("shared_model", "optimizer")

    def update(self, statevar):
        assert self.t_start < self.t

        n = self.t - self.t_start

        self.assert_shared_memory()

        if statevar is None:
            R = 0
        else:
            with torch.no_grad(), pfrl.utils.evaluating(self.model):
                if self.recurrent:
                    (_, vout), _ = one_step_forward(
                        self.model, statevar, self.train_recurrent_states
                    )
                else:
                    _, vout = self.model(statevar)
            R = float(vout)

        pi_loss_factor = self.pi_loss_coef
        v_loss_factor = self.v_loss_coef

        # Normalize the loss of sequences truncated by terminal states
        if self.keep_loss_scale_same and self.t - self.t_start < self.t_max:
            factor = self.t_max / (self.t - self.t_start)
            pi_loss_factor *= factor
            v_loss_factor *= factor

        if self.normalize_grad_by_t_max:
            pi_loss_factor /= self.t - self.t_start
            v_loss_factor /= self.t - self.t_start

        # Batch re-compute for efficient backprop
        batch_obs = self.batch_states(
            [self.past_obs[i] for i in range(self.t_start, self.t)],
            self.device,
            self.phi,
        )
        if self.recurrent:
            (batch_distrib, batch_v), _ = pack_and_forward(
                self.model,
                [batch_obs],
                self.past_recurrent_state[self.t_start],
            )
        else:
            batch_distrib, batch_v = self.model(batch_obs)
        batch_action = torch.stack(
            [self.past_action[i] for i in range(self.t_start, self.t)]
        )
        batch_log_prob = batch_distrib.log_prob(batch_action)
        batch_entropy = batch_distrib.entropy()
        rev_returns = []
        for i in reversed(range(self.t_start, self.t)):
            R *= self.gamma
            R += self.past_rewards[i]
            rev_returns.append(R)
        batch_return = torch.as_tensor(list(reversed(rev_returns)), dtype=torch.float)
        batch_adv = batch_return - batch_v.detach().squeeze(-1)
        assert batch_log_prob.shape == (n,)
        assert batch_adv.shape == (n,)
        assert batch_entropy.shape == (n,)
        pi_loss = torch.sum(
            -batch_adv * batch_log_prob - self.beta * batch_entropy, dim=0
        )
        assert batch_v.shape == (n, 1)
        assert batch_return.shape == (n,)
        v_loss = F.mse_loss(batch_v, batch_return[..., None], reduction="sum") / 2

        if pi_loss_factor != 1.0:
            pi_loss *= pi_loss_factor

        if v_loss_factor != 1.0:
            v_loss *= v_loss_factor

        if self.process_idx == 0:
            logger.debug("pi_loss:%s v_loss:%s", pi_loss, v_loss)

        total_loss = torch.squeeze(pi_loss) + torch.squeeze(v_loss)

        # Compute gradients using thread-specific model
        self.model.zero_grad()
        total_loss.backward()
        if self.max_grad_norm is not None:
            clip_l2_grad_norm_(self.model.parameters(), self.max_grad_norm)
        # Copy the gradients to the globally shared model
        copy_param.copy_grad(target_link=self.shared_model, source_link=self.model)
        # Update the globally shared model
        self.optimizer.step()
        if self.process_idx == 0:
            logger.debug("update")

        self.sync_parameters()

        self.past_obs = {}
        self.past_action = {}
        self.past_rewards = {}
        self.past_recurrent_state = {}

        self.t_start = self.t

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

        self.past_obs[self.t] = obs

        with torch.no_grad():
            statevar = self.batch_states([obs], self.device, self.phi)
            if self.recurrent:
                self.past_recurrent_state[self.t] = self.train_recurrent_states
                (pout, vout), self.train_recurrent_states = one_step_forward(
                    self.model, statevar, self.train_recurrent_states
                )
            else:
                pout, vout = self.model(statevar)
            # Do not backprop through sampled actions
            action = pout.sample()
            self.past_action[self.t] = action[0].detach()
            action = action.cpu().numpy()[0]

        # Update stats
        self.average_value += (1 - self.average_value_decay) * (
            float(vout) - self.average_value
        )
        self.average_entropy += (1 - self.average_entropy_decay) * (
            float(pout.entropy()) - self.average_entropy
        )

        return action

    def _observe_train(self, obs, reward, done, reset):
        self.t += 1
        self.past_rewards[self.t - 1] = reward
        if self.process_idx == 0:
            logger.debug(
                "t:%s action:%s reward:%s", self.t, self.past_action[self.t - 1], reward
            )
        if self.t - self.t_start == self.t_max or done or reset:
            if done:
                statevar = None
            else:
                statevar = self.batch_states([obs], self.device, self.phi)
            self.update(statevar)
        if done or reset:
            self.train_recurrent_states = None

    def _act_eval(self, obs):
        # Use the process-local model for acting
        with torch.no_grad(), pfrl.utils.evaluating(self.model):
            statevar = self.batch_states([obs], self.device, self.phi)
            if self.recurrent:
                (pout, _), self.test_recurrent_states = one_step_forward(
                    self.model, statevar, self.test_recurrent_states
                )
            else:
                pout, _ = self.model(statevar)
            if self.act_deterministically:
                return mode_of_distribution(pout).cpu().numpy()[0]
            else:
                return pout.sample().cpu().numpy()[0]

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
        ]
