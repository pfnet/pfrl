import collections

from pfrl.collections.prioritized import PrioritizedBuffer
from pfrl.collections.random_access_queue import RandomAccessQueue
from pfrl.replay_buffer import random_subseq
from pfrl.replay_buffers import EpisodicReplayBuffer, PriorityWeightError


class PrioritizedEpisodicReplayBuffer(EpisodicReplayBuffer, PriorityWeightError):
    def __init__(
        self,
        capacity=None,
        alpha=0.6,
        beta0=0.4,
        betasteps=2e5,
        eps=1e-8,
        normalize_by_max=True,
        default_priority_func=None,
        uniform_ratio=0,
        wait_priority_after_sampling=True,
        return_sample_weights=True,
        error_min=None,
        error_max=None,
    ):
        self.current_episode = collections.defaultdict(list)
        self.episodic_memory = PrioritizedBuffer(
            capacity=None, wait_priority_after_sampling=wait_priority_after_sampling
        )
        self.memory = RandomAccessQueue(maxlen=capacity)
        self.capacity_left = capacity
        self.default_priority_func = default_priority_func
        self.uniform_ratio = uniform_ratio
        self.return_sample_weights = return_sample_weights
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

    def sample_episodes(self, n_episodes, max_len=None):
        """Sample n unique samples from this replay buffer"""
        assert len(self.episodic_memory) >= n_episodes
        episodes, probabilities, min_prob = self.episodic_memory.sample(
            n_episodes, uniform_ratio=self.uniform_ratio
        )
        if max_len is not None:
            episodes = [random_subseq(ep, max_len) for ep in episodes]
        if self.return_sample_weights:
            weights = self.weights_from_probabilities(probabilities, min_prob)
            return episodes, weights
        else:
            return episodes

    def update_errors(self, errors):
        self.episodic_memory.set_last_priority(self.priority_from_errors(errors))

    def stop_current_episode(self, env_id=0):
        current_episode = self.current_episode[env_id]
        if current_episode:
            if self.default_priority_func is not None:
                priority = self.default_priority_func(current_episode)
            else:
                priority = None
            self.memory.extend(current_episode)
            self.episodic_memory.append(current_episode, priority=priority)
            if self.capacity_left is not None:
                self.capacity_left -= len(current_episode)
            self.current_episode[env_id] = []
            while self.capacity_left is not None and self.capacity_left < 0:
                discarded_episode = self.episodic_memory.popleft()
                self.capacity_left += len(discarded_episode)
        assert not self.current_episode[env_id]
