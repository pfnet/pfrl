from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Optional

import numpy as np
import torch

from pfrl.utils.batch_states import batch_states
from pfrl.utils.recurrent import (
    concatenate_recurrent_states,
    flatten_sequences_time_first,
    recurrent_state_from_numpy,
)


class AbstractReplayBuffer(object, metaclass=ABCMeta):
    """Defines a common interface of replay buffer.

    You can append transitions to the replay buffer and later sample from it.
    Replay buffers are typically used in experience replay.
    """

    @abstractmethod
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
        """Append a transition to this replay buffer.

        Args:
            state: s_t
            action: a_t
            reward: r_t
            next_state: s_{t+1} (can be None if terminal)
            next_action: a_{t+1} (can be None for off-policy algorithms)
            is_state_terminal (bool)
            env_id (object): Object that is unique to each env. It indicates
                which env a given transition came from in multi-env training.
            **kwargs: Any other information to store.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, n):
        """Sample n unique transitions from this replay buffer.

        Args:
            n (int): Number of transitions to sample.
        Returns:
            Sequence of n sampled transitions.
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        """Return the number of transitions in the buffer.

        Returns:
            Number of transitions in the buffer.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, filename):
        """Save the content of the buffer to a file.

        Args:
            filename (str): Path to a file.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, filename):
        """Load the content of the buffer from a file.

        Args:
            filename (str): Path to a file.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def capacity(self) -> Optional[int]:
        """Returns the capacity of the buffer in number of transitions.

        If unbounded, returns None instead.
        """
        raise NotImplementedError

    @abstractmethod
    def stop_current_episode(self, env_id=0):
        """Notify the buffer that the current episode is interrupted.

        You may want to interrupt the current episode and start a new one
        before observing a terminal state. This is typical in continuing envs.
        In such cases, you need to call this method before appending a new
        transition so that the buffer will treat it as an initial transition of
        a new episode.

        This method should not be called after an episode whose termination is
        already notified by appending a transition with is_state_terminal=True.

        Args:
            env_id (object): Object that is unique to each env. It indicates
                which env's current episode is interrupted in multi-env
                training.
        """
        raise NotImplementedError


class AbstractEpisodicReplayBuffer(AbstractReplayBuffer):
    """Defines a common interface of episodic replay buffer.

    Episodic replay buffers allows you to append and sample episodes.
    """

    @abstractmethod
    def sample_episodes(self, n_episodes, max_len=None):
        """Sample n unique (sub)episodes from this replay buffer.

        Args:
            n (int): Number of episodes to sample.
            max_len (int or None): Maximum length of sampled episodes. If it is
                smaller than the length of some episode, the subsequence of the
                episode is sampled instead. If None, full episodes are always
                returned.
        Returns:
            Sequence of n sampled episodes, each of which is a sequence of
            transitions.
        """
        raise NotImplementedError

    @abstractproperty
    def n_episodes(self):
        """Returns the number of episodes in the buffer.

        Returns:
            Number of episodes in the buffer.
        """
        raise NotImplementedError


def random_subseq(seq, subseq_len):
    if len(seq) <= subseq_len:
        return seq
    else:
        i = np.random.randint(0, len(seq) - subseq_len + 1)
        return seq[i : i + subseq_len]


def batch_experiences(experiences, device, phi, gamma, batch_states=batch_states):
    """Takes a batch of k experiences each of which contains j

    consecutive transitions and vectorizes them, where j is between 1 and n.

    Args:
        experiences: list of experiences. Each experience is a list
            containing between 1 and n dicts containing
              - state (object): State
              - action (object): Action
              - reward (float): Reward
              - is_state_terminal (bool): True iff next state is terminal
              - next_state (object): Next state
        device : GPU or CPU the tensor should be placed on
        phi : Preprocessing function
        gamma: discount factor
        batch_states: function that converts a list to a batch
    Returns:
        dict of batched transitions
    """

    batch_exp = {
        "state": batch_states([elem[0]["state"] for elem in experiences], device, phi),
        "action": torch.as_tensor(
            [elem[0]["action"] for elem in experiences], device=device
        ),
        "reward": torch.as_tensor(
            [
                sum((gamma ** i) * exp[i]["reward"] for i in range(len(exp)))
                for exp in experiences
            ],
            dtype=torch.float32,
            device=device,
        ),
        "next_state": batch_states(
            [elem[-1]["next_state"] for elem in experiences], device, phi
        ),
        "is_state_terminal": torch.as_tensor(
            [
                any(transition["is_state_terminal"] for transition in exp)
                for exp in experiences
            ],
            dtype=torch.float32,
            device=device,
        ),
        "discount": torch.as_tensor(
            [(gamma ** len(elem)) for elem in experiences],
            dtype=torch.float32,
            device=device,
        ),
    }
    if all(elem[-1]["next_action"] is not None for elem in experiences):
        batch_exp["next_action"] = torch.as_tensor(
            [elem[-1]["next_action"] for elem in experiences], device=device
        )
    return batch_exp


def _is_sorted_desc_by_lengths(lst):
    return all(len(a) >= len(b) for a, b in zip(lst, lst[1:]))


def batch_recurrent_experiences(
    experiences, device, phi, gamma, batch_states=batch_states
):
    """Batch experiences for recurrent model updates.

    Args:
        experiences: list of episodes. Each episode is a list
            containing between 1 and n dicts, each containing:
              - state (object): State
              - action (object): Action
              - reward (float): Reward
              - is_state_terminal (bool): True iff next state is terminal
              - next_state (object): Next state
            The list must be sorted desc by lengths to be packed by
            `torch.nn.rnn.pack_sequence` later.
        device : GPU or CPU the tensor should be placed on
        phi : Preprocessing function
        gamma: discount factor
        batch_states: function that converts a list to a batch
    Returns:
        dict of batched transitions
    """
    assert _is_sorted_desc_by_lengths(experiences)
    flat_transitions = flatten_sequences_time_first(experiences)
    batch_exp = {
        "state": [
            batch_states([transition["state"] for transition in ep], device, phi)
            for ep in experiences
        ],
        "action": torch.as_tensor(
            [transition["action"] for transition in flat_transitions], device=device
        ),
        "reward": torch.as_tensor(
            [transition["reward"] for transition in flat_transitions],
            dtype=torch.float,
            device=device,
        ),
        "next_state": [
            batch_states([transition["next_state"] for transition in ep], device, phi)
            for ep in experiences
        ],
        "is_state_terminal": torch.as_tensor(
            [transition["is_state_terminal"] for transition in flat_transitions],
            dtype=torch.float,
            device=device,
        ),
        "discount": torch.full(
            (len(flat_transitions),), gamma, dtype=torch.float, device=device
        ),
        "recurrent_state": recurrent_state_from_numpy(
            concatenate_recurrent_states(
                [ep[0]["recurrent_state"] for ep in experiences]
            ),
            device,
        ),
        "next_recurrent_state": recurrent_state_from_numpy(
            concatenate_recurrent_states(
                [ep[0]["next_recurrent_state"] for ep in experiences]
            ),
            device,
        ),
    }
    # Batch next actions only when all the transitions have them
    if all(transition["next_action"] is not None for transition in flat_transitions):
        batch_exp["next_action"] = torch.as_tensor(
            [transition["next_action"] for transition in flat_transitions],
            device=device,
        )
    return batch_exp


class ReplayUpdater(object):
    """Object that handles update schedule and configurations.

    Args:
        replay_buffer (ReplayBuffer): Replay buffer
        update_func (callable): Callable that accepts one of these:
            (1) a list of transition dicts (if episodic_update=False)
            (2) a list of lists of transition dicts (if episodic_update=True)
        replay_start_size (int): if the replay buffer's size is less than
            replay_start_size, skip update
        batchsize (int): Minibatch size
        update_interval (int): Model update interval in step
        n_times_update (int): Number of repetition of update
        episodic_update (bool): Use full episodes for update if set True
        episodic_update_len (int or None): Subsequences of this length are used
            for update if set int and episodic_update=True
    """

    def __init__(
        self,
        replay_buffer,
        update_func,
        batchsize,
        episodic_update,
        n_times_update,
        replay_start_size,
        update_interval,
        episodic_update_len=None,
    ):

        assert batchsize <= replay_start_size
        self.replay_buffer = replay_buffer
        self.update_func = update_func
        self.batchsize = batchsize
        self.episodic_update = episodic_update
        self.episodic_update_len = episodic_update_len
        self.n_times_update = n_times_update
        self.replay_start_size = replay_start_size
        self.update_interval = update_interval

    def update_if_necessary(self, iteration):
        """Update the model if the condition is met.

        Args:
            iteration (int): Timestep.

        Returns:
            bool: True iff the condition was updated this time.
        """
        if len(self.replay_buffer) < self.replay_start_size:
            return False

        if self.episodic_update and self.replay_buffer.n_episodes < self.batchsize:
            return False

        if iteration % self.update_interval != 0:
            return False

        for _ in range(self.n_times_update):
            if self.episodic_update:
                episodes = self.replay_buffer.sample_episodes(
                    self.batchsize, self.episodic_update_len
                )
                self.update_func(episodes)
            else:
                transitions = self.replay_buffer.sample(self.batchsize)
                self.update_func(transitions)
        return True
