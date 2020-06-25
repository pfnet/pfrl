import collections

import numpy as np

from pfrl.collections.prioritized import PrioritizedBuffer
from pfrl.replay_buffers.replay_buffer import ReplayBuffer  # NOQA


class PriorityWeightError(object):
    """For proportional prioritization

    alpha determines how much prioritization is used.

    beta determines how much importance sampling weights are used. beta is
    scheduled by ``beta0`` and ``betasteps``.

    Args:
        alpha (float): Exponent of errors to compute probabilities to sample
        beta0 (float): Initial value of beta
        betasteps (float): Steps to anneal beta to 1
        eps (float): To revisit a step after its error becomes near zero
        normalize_by_max (str): Method to normalize weights. ``'batch'`` or
            ``True`` (default): divide by the maximum weight in the sampled
            batch. ``'memory'``: divide by the maximum weight in the memory.
            ``False``: do not normalize.
    """

    def __init__(
        self, alpha, beta0, betasteps, eps, normalize_by_max, error_min, error_max
    ):
        assert 0.0 <= alpha
        assert 0.0 <= beta0 <= 1.0
        self.alpha = alpha
        self.beta = beta0
        if betasteps is None:
            self.beta_add = 0
        else:
            self.beta_add = (1.0 - beta0) / betasteps
        self.eps = eps
        if normalize_by_max is True:
            normalize_by_max = "batch"
        assert normalize_by_max in [False, "batch", "memory"]
        self.normalize_by_max = normalize_by_max
        self.error_min = error_min
        self.error_max = error_max

    def priority_from_errors(self, errors):
        def _clip_error(error):
            if self.error_min is not None:
                error = max(self.error_min, error)
            if self.error_max is not None:
                error = min(self.error_max, error)
            return error

        return [(_clip_error(d) + self.eps) ** self.alpha for d in errors]

    def weights_from_probabilities(self, probabilities, min_probability):
        if self.normalize_by_max == "batch":
            # discard global min and compute batch min
            min_probability = np.min(probabilities)
        if self.normalize_by_max:
            weights = [(p / min_probability) ** -self.beta for p in probabilities]
        else:
            weights = [(len(self.memory) * p) ** -self.beta for p in probabilities]
        self.beta = min(1.0, self.beta + self.beta_add)
        return weights


class PrioritizedReplayBuffer(ReplayBuffer, PriorityWeightError):
    """Stochastic Prioritization

    https://arxiv.org/pdf/1511.05952.pdf Section 3.3
    proportional prioritization

    Args:
        capacity (int): capacity in terms of number of transitions
        alpha (float): Exponent of errors to compute probabilities to sample
        beta0 (float): Initial value of beta
        betasteps (int): Steps to anneal beta to 1
        eps (float): To revisit a step after its error becomes near zero
        normalize_by_max (bool): Method to normalize weights. ``'batch'`` or
            ``True`` (default): divide by the maximum weight in the sampled
            batch. ``'memory'``: divide by the maximum weight in the memory.
            ``False``: do not normalize
    """

    def __init__(
        self,
        capacity=None,
        alpha=0.6,
        beta0=0.4,
        betasteps=2e5,
        eps=0.01,
        normalize_by_max=True,
        error_min=0,
        error_max=1,
        num_steps=1,
    ):
        self.capacity = capacity
        assert num_steps > 0
        self.num_steps = num_steps
        self.memory = PrioritizedBuffer(capacity=capacity)
        self.last_n_transitions = collections.defaultdict(
            lambda: collections.deque([], maxlen=num_steps)
        )
        PriorityWeightError.__init__(
            self,
            alpha,
            beta0,
            betasteps,
            eps,
            normalize_by_max,
            error_min=error_min,
            error_max=error_max,
        )

    def sample(self, n):
        assert len(self.memory) >= n
        sampled, probabilities, min_prob = self.memory.sample(n)
        weights = self.weights_from_probabilities(probabilities, min_prob)
        for e, w in zip(sampled, weights):
            e[0]["weight"] = w
        return sampled

    def update_errors(self, errors):
        self.memory.set_last_priority(self.priority_from_errors(errors))
