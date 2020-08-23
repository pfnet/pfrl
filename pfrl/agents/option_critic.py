# Adapted from https://github.com/lweitkamp/option-critic-pytorch

import torch
import torch.nn as nn
from torch.distributions import Categorical, Bernoulli

class OptionCriticNetwork(nn.Module):
    def __init__(self, featureNetwork, terminationNetwork, QNetwork, feature_output_size, device='cpu',
                 eps_start = 1.0, eps_min = 0.1, eps_decay = int(1e6)):
        super(OptionCriticNetwork, self).__init__()

        self.eps_start = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.num_steps = 0

        self.features = featureNetwork
        self.terminations = terminationNetwork
        self.Q = QNetwork
        self.options_W = nn.Parameter(torch.zeros(num_options, feature_output_size, num_actions))
        self.options_b = nn.Parameter(torch.zeros(num_options, num_actions))

        self.to(device)

    def get_state(self, obs):
        obs = obs.to(self.device)
        state = self.features(obs)
        return state

    def get_Q(self, state):
        return self.Q(state)

    def predict_option_termination(self, state, current_option):
        termination = self.terminations(state)[:, current_option].sigmoid()
        option_termination = Bernoulli(termination).sample()

        Q = self.get_Q(state)
        next_option = Q.argmax(dim=-1)
        return bool(option_termination.item()), next_option.item()

    def get_terminations(self, state):
        return self.terminations(state).sigmoid()

    def get_action(self, state, option):
        logits = state @ self.options_W[option] + self.options_b[option]
        action_dist = logits.softmax(dim=-1)
        action_dist = Categorical(action_dist)

        action = action_dist.sample()
        logp = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        return action.item(), logp, entropy

    def greedy_option(self, state):
        Q = self.get_Q(state)
        return Q.argmax(dim=-1).item()

    @property
    def epsilon(self):
        eps = self.eps_min + (self.eps_start - self.eps_min) * exp(-self.num_steps / self.eps_decay)
        self.num_steps += 1

        return eps
