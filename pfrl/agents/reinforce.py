import warnings
from logging import getLogger

import numpy as np
import torch

import pfrl
from pfrl import agent
from pfrl.utils import clip_l2_grad_norm_
from pfrl.utils.mode_of_distribution import mode_of_distribution
from pfrl.utils.recurrent import one_step_forward


class REINFORCE(agent.AttributeSavingMixin, agent.Agent):
    """William's episodic REINFORCE.

    Args:
        model (Policy): Model to train. It must be a callable that accepts
            observations as input and return action distributions
            (Distribution).
        optimizer (torch.optim.Optimizer): optimizer used to train the model
        gpu (int): GPU device id if not None nor negative
        beta (float): Weight coefficient for the entropy regularizaiton term.
        phi (callable): Feature extractor function
        act_deterministically (bool): If set true, choose most probable actions
            in act method.
        batchsize (int): Number of episodes used for each update
        backward_separately (bool): If set true, call backward separately for
            each episode and accumulate only gradients.
        average_entropy_decay (float): Decay rate of average entropy. Used only
            to record statistics.
        batch_states (callable): Method which makes a batch of observations.
            default is `pfrl.utils.batch_states`
        recurrent (bool): If set to True, `model` is assumed to implement
            `pfrl.nn.Recurrent` and update in a recurrent
            manner.
        max_grad_norm (float or None): Maximum L2 norm of the gradient used for
            gradient clipping. If set to None, the gradient is not clipped.
        logger (logging.Logger): Logger to be used.
    """

    saved_attributes = ("model", "optimizer")

    def __init__(
        self,
        model,
        optimizer,
        gpu=None,
        beta=0,
        phi=lambda x: x,
        batchsize=1,
        act_deterministically=False,
        average_entropy_decay=0.999,
        backward_separately=False,
        batch_states=pfrl.utils.batch_states,
        recurrent=False,
        max_grad_norm=None,
        logger=None,
    ):

        self.model = model
        if gpu is not None and gpu >= 0:
            assert torch.cuda.is_available()
            self.device = torch.device("cuda:{}".format(gpu))
            self.model.to(self.device)
        else:
            self.device = torch.device("cpu")
        self.optimizer = optimizer
        self.beta = beta
        self.phi = phi
        self.batchsize = batchsize
        self.backward_separately = backward_separately
        self.act_deterministically = act_deterministically
        self.average_entropy_decay = average_entropy_decay
        self.batch_states = batch_states
        self.recurrent = recurrent
        self.max_grad_norm = max_grad_norm
        self.logger = logger or getLogger(__name__)

        # Recurrent states of the model
        self.train_recurrent_states = None
        self.test_recurrent_states = None

        # Statistics
        self.average_entropy = 0

        self.t = 0
        self.reward_sequences = [[]]
        self.log_prob_sequences = [[]]
        self.entropy_sequences = [[]]
        self.n_backward = 0

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

        batch_obs = self.batch_states([obs], self.device, self.phi)
        if self.recurrent:
            action_distrib, self.train_recurrent_states = one_step_forward(
                self.model, batch_obs, self.train_recurrent_states
            )
        else:
            action_distrib = self.model(batch_obs)
        batch_action = action_distrib.sample()

        # Save values used to compute losses
        self.log_prob_sequences[-1].append(action_distrib.log_prob(batch_action))
        self.entropy_sequences[-1].append(action_distrib.entropy())

        action = batch_action.cpu().numpy()[0]

        self.logger.debug("t:%s a:%s", self.t, action)

        # Update stats
        self.average_entropy += (1 - self.average_entropy_decay) * (
            float(action_distrib.entropy()) - self.average_entropy
        )

        return action

    def _observe_train(self, obs, reward, done, reset):
        self.reward_sequences[-1].append(reward)
        self.t += 1

        if done or reset:
            if not done:
                warnings.warn(
                    "Since REINFORCE supports episodic environments only, "
                    "reset=True with done=False will throw away the last episode."
                )
                self.reward_sequences[-1] = []
                self.log_prob_sequences[-1] = []
                self.entropy_sequences[-1] = []
            elif done:
                self.reward_sequences[-1].append(reward)
                if self.backward_separately:
                    self.accumulate_grad()
                    if self.n_backward == self.batchsize:
                        self.update_with_accumulated_grad()
                else:
                    if len(self.reward_sequences) == self.batchsize:
                        self.batch_update()
                    else:
                        # Prepare for the next episode
                        self.reward_sequences.append([])
                        self.log_prob_sequences.append([])
                        self.entropy_sequences.append([])
            self.train_recurrent_states = None

    def _act_eval(self, obs):
        with torch.no_grad():
            batch_obs = self.batch_states([obs], self.device, self.phi)
            if self.recurrent:
                action_distrib, self.test_recurrent_states = one_step_forward(
                    self.model, batch_obs, self.test_recurrent_states
                )
            else:
                action_distrib = self.model(batch_obs)
            if self.act_deterministically:
                return mode_of_distribution(action_distrib).cpu().numpy()[0]
            else:
                return action_distrib.sample().cpu().numpy()[0]

    def _observe_eval(self, obs, reward, done, reset):
        if done or reset:
            self.test_recurrent_states = None

    def accumulate_grad(self):
        if self.n_backward == 0:
            self.optimizer.zero_grad()
        # Compute losses
        losses = []
        for r_seq, log_prob_seq, ent_seq in zip(
            self.reward_sequences, self.log_prob_sequences, self.entropy_sequences
        ):
            assert len(r_seq) - 1 == len(log_prob_seq) == len(ent_seq)
            # Convert rewards into returns (=sum of future rewards)
            R_seq = np.cumsum(list(reversed(r_seq[1:])))[::-1]
            for R, log_prob, entropy in zip(R_seq, log_prob_seq, ent_seq):
                loss = -R * log_prob - self.beta * entropy
                losses.append(loss)
        total_loss = torch.stack(losses).sum() / self.batchsize
        total_loss.backward()
        self.reward_sequences = [[]]
        self.log_prob_sequences = [[]]
        self.entropy_sequences = [[]]
        self.n_backward += 1

    def batch_update(self):
        assert len(self.reward_sequences) == self.batchsize
        assert len(self.log_prob_sequences) == self.batchsize
        assert len(self.entropy_sequences) == self.batchsize
        # Update the model
        assert self.n_backward == 0
        self.accumulate_grad()
        if self.max_grad_norm is not None:
            clip_l2_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.n_backward = 0

    def update_with_accumulated_grad(self):
        assert self.n_backward == self.batchsize
        if self.max_grad_norm is not None:
            clip_l2_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.n_backward = 0

    def get_statistics(self):
        return [
            ("average_entropy", self.average_entropy),
        ]
