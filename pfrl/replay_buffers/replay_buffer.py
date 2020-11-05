import collections
import pickle
from typing import Optional

from pfrl import replay_buffer
from pfrl.collections.random_access_queue import RandomAccessQueue


class ReplayBuffer(replay_buffer.AbstractReplayBuffer):
    """Experience Replay Buffer

    As described in
    https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf.

    Args:
        capacity (int): capacity in terms of number of transitions
        num_steps (int): Number of timesteps per stored transition
            (for N-step updates)
    """

    # Implements AbstractReplayBuffer.capacity
    capacity: Optional[int] = None

    def __init__(self, capacity: Optional[int] = None, num_steps: int = 1):
        self.capacity = capacity
        assert num_steps > 0
        self.num_steps = num_steps
        self.memory = RandomAccessQueue(maxlen=capacity)
        self.last_n_transitions: collections.defaultdict = collections.defaultdict(
            lambda: collections.deque([], maxlen=num_steps)
        )

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
        last_n_transitions = self.last_n_transitions[env_id]
        experience = dict(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            next_action=next_action,
            is_state_terminal=is_state_terminal,
            **kwargs
        )
        last_n_transitions.append(experience)
        if is_state_terminal:
            while last_n_transitions:
                self.memory.append(list(last_n_transitions))
                del last_n_transitions[0]
            assert len(last_n_transitions) == 0
        else:
            if len(last_n_transitions) == self.num_steps:
                self.memory.append(list(last_n_transitions))

    def stop_current_episode(self, env_id=0):
        last_n_transitions = self.last_n_transitions[env_id]
        # if n-step transition hist is not full, add transition;
        # if n-step hist is indeed full, transition has already been added;
        if 0 < len(last_n_transitions) < self.num_steps:
            self.memory.append(list(last_n_transitions))
        # avoid duplicate entry
        if 0 < len(last_n_transitions) <= self.num_steps:
            del last_n_transitions[0]
        while last_n_transitions:
            self.memory.append(list(last_n_transitions))
            del last_n_transitions[0]
        assert len(last_n_transitions) == 0

    def sample(self, num_experiences):
        assert len(self.memory) >= num_experiences
        return self.memory.sample(num_experiences)

    def __len__(self):
        return len(self.memory)

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.memory, f)

    def load(self, filename):
        with open(filename, "rb") as f:
            self.memory = pickle.load(f)
        if isinstance(self.memory, collections.deque):
            # Load v0.2
            self.memory = RandomAccessQueue(self.memory, maxlen=self.memory.maxlen)
