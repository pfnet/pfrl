import copy

import numpy as np

from pfrl.replay_buffers.episodic import EpisodicReplayBuffer
from pfrl.replay_buffer import random_subseq


def relabel_transition_goal(transition, goal_transition,
                            reward_fn, swap_keys_list):
    # Relabel/replace the desired goal for the transition with new_goal
    for desired_obs_key, achieved_obs_key in swap_keys_list:
        replacement = goal_transition["next_state"][achieved_obs_key]
        transition["state"][desired_obs_key] = replacement
        transition["next_state"][desired_obs_key] = replacement
    new_goal = goal_transition["next_state"]["achieved_goal"]
    achieved_goal = transition["next_state"]["achieved_goal"]
    transition["reward"] = reward_fn(new_goal, achieved_goal)
    return transition


class HindsightReplayStrategy():
    """ReplayStrategy for Hindsight experience replay
    """

    def __init__(self, reward_fn):
        self.reward_fn = reward_fn

    def apply(self, episodes,  reward_fn, swap_keys_list):
        return episodes

class ReplayFinalGoal(HindsightReplayStrategy):
    """Replay final goal.
    """

    def apply(self, episodes, reward_fn, swap_keys_list):
        batch_size = len(episodes)
        episode_lens = np.array([len(episode) for episode in episodes])

        # Randomly select time-steps from each episode
        ts = [np.random.randint(ep_len) for ep_len in episode_lens]
        ts = np.array(ts)

        # Select subset for hindsight goal replacement.
        apply_hers = np.random.uniform(size=batch_size) < 0.5

        batch = []
        for episode, apply_her, t in zip(episodes, apply_hers, ts):
            transition = episode[t]
            if apply_her:
                final_transition = episode[-1]
                final_goal = final_transition["next_state"]["achieved_goal"]
                transition = copy.deepcopy(transition)
                transition = relabel_transition_goal(
                    transition, final_transition, reward_fn, swap_keys_list)
            batch.append([transition])
        return batch

class ReplayFutureGoal(HindsightReplayStrategy):
    """Replay random future goal.

        Args:
            ignore_null_goals (bool): no replace with goal when nothing achieved
            future_k (int): number of future goals to sample per true sample
    """

    def __init__(self, future_k=4):
        self.future_prob = 1.0 - 1.0 / (float(future_k) + 1)

    def apply(self, episodes, reward_fn, swap_keys_list):
        """Sample with the future strategy
        """
        batch_size = len(episodes)
        episode_lens = np.array([len(episode) for episode in episodes])

        # Randomly select time-steps from each episode
        ts = [np.random.randint(ep_len) for ep_len in episode_lens]
        ts = np.array(ts)

        # Select subset for hindsight goal replacement. future_k controls ratio
        apply_hers = np.random.uniform(size=batch_size) < self.future_prob

        # Randomly select offsets for future goals
        future_offset = np.random.uniform(
            size=batch_size) * (episode_lens - ts)
        future_offset = future_offset.astype(int)
        future_ts = ts + future_offset
        batch = []
        for episode, apply_her, t, future_t in zip(episodes,
                                                   apply_hers,
                                                   ts, future_ts):
            transition = episode[t]
            if apply_her:
                future_transition = episode[future_t]
                future_goal = future_transition["next_state"]["achieved_goal"]
                transition = copy.deepcopy(transition)
                transition = relabel_transition_goal(
                    transition, future_transition, reward_fn, swap_keys_list)
            batch.append([transition])
        return batch

class HindsightReplayBuffer(EpisodicReplayBuffer):
    """Hindsight Replay Buffer

     https://arxiv.org/abs/1707.01495
     We currently do not support N-step transitions for the
     Hindsight Buffer.
     Args:
        reward_fn(fn): Calculate reward from achieved & observed goals
        replay_strategy: instance of HindsightReplayStrategy()
        capacity (int): Capacity of the replay buffer
        swap_list (list): a list of tuples of keys to swap in the
            observation. E.g. [(("desired_x", "achieved_x"))] This is used
            to replace a transition's "desired_x" with a goal transition's
            "achieved_x"
    """

    def __init__(self,
                 reward_fn,
                 replay_strategy,
                 capacity=None,
                 swap_list=[('desired_goal', 'achieved_goal')]):

        assert replay_strategy is not None
        self.reward_fn = reward_fn
        self.replay_strategy = replay_strategy
        self.swap_keys_list = swap_list
        assert ('desired_goal', 'achieved_goal') in self.swap_keys_list

        super(HindsightReplayBuffer, self).__init__(capacity)
        # probability of sampling a future goal instead of a true goal


    def sample(self, n):
        # Sample n transitions from the hindsight replay buffer
        assert len(self.memory) >= n
        # Select n episodes
        episodes = self.sample_episodes(n)
        batch = self.replay_strategy.apply(episodes,
                                           self.reward_fn,
                                           self.swap_keys_list)
        return batch

    def sample_episodes(self, n_episodes, max_len=None):
        episodes = self.sample_with_replacement(n_episodes)
        if max_len is not None:
            return [random_subseq(ep, max_len) for ep in episodes]
        else:
            return episodes

    def sample_with_replacement(self, k):
        return [self.episodic_memory[i] for i in
                np.random.randint(0, len(self.episodic_memory), k)]