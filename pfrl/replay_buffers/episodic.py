import collections
import pickle
from typing import Optional

from pfrl.collections.random_access_queue import RandomAccessQueue
from pfrl.replay_buffer import AbstractEpisodicReplayBuffer, random_subseq


class EpisodicReplayBuffer(AbstractEpisodicReplayBuffer):

    # Implements AbstractReplayBuffer.capacity
    capacity: Optional[int] = None

    def __init__(self, capacity=None):
        self.current_episode = collections.defaultdict(list)
        self.episodic_memory = RandomAccessQueue()
        self.memory = RandomAccessQueue()
        self.capacity = capacity

    def append(
        self,
        state,
        action,
        reward,
        next_state=None,
        next_action=None,
        is_state_terminal=False,
        env_id=0,
        **kwargs
    ):
        current_episode = self.current_episode[env_id]
        experience = dict(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            next_action=next_action,
            is_state_terminal=is_state_terminal,
            **kwargs
        )
        current_episode.append(experience)
        if is_state_terminal:
            self.stop_current_episode(env_id=env_id)

    def sample(self, n):
        assert len(self.memory) >= n
        return self.memory.sample(n)

    def sample_episodes(self, n_episodes, max_len=None):
        assert len(self.episodic_memory) >= n_episodes
        episodes = self.episodic_memory.sample(n_episodes)
        if max_len is not None:
            return [random_subseq(ep, max_len) for ep in episodes]
        else:
            return episodes

    def __len__(self):
        return len(self.memory)

    @property
    def n_episodes(self):
        return len(self.episodic_memory)

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump((self.memory, self.episodic_memory), f)

    def load(self, filename):
        with open(filename, "rb") as f:
            memory = pickle.load(f)
        if isinstance(memory, tuple):
            self.memory, self.episodic_memory = memory
        else:
            # Load v0.2
            # FIXME: The code works with EpisodicReplayBuffer
            # but not with PrioritizedEpisodicReplayBuffer
            self.memory = RandomAccessQueue(memory)
            self.episodic_memory = RandomAccessQueue()

            # Recover episodic_memory with best effort.
            episode = []
            for item in self.memory:
                episode.append(item)
                if item["is_state_terminal"]:
                    self.episodic_memory.append(episode)
                    episode = []

    def stop_current_episode(self, env_id=0):
        current_episode = self.current_episode[env_id]
        if current_episode:
            self.episodic_memory.append(current_episode)
            for transition in current_episode:
                self.memory.append([transition])
            self.current_episode[env_id] = []
            while self.capacity is not None and len(self.memory) > self.capacity:
                discarded_episode = self.episodic_memory.popleft()
                for _ in range(len(discarded_episode)):
                    self.memory.popleft()
        assert not self.current_episode[env_id]
