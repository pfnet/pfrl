import torch
import numpy as np
from pfrl.agent import HRLAgent
from pfrl.utils import Subgoal
from pfrl.replay_buffers import (
    LowerControllerReplayBuffer,
    HigherControllerReplayBuffer
)


def _is_update(episode, freq, ignore=0, rem=0):
    if episode != ignore and episode % freq == rem:
        return True
    return False


class HIROAgent(HRLAgent):
    def __init__(self,
                 state_dim,
                 action_dim,
                 goal_dim,
                 subgoal_dim,
                 scale_low,
                 start_training_steps,
                 model_save_freq,
                 model_path,
                 buffer_size,
                 batch_size,
                 buffer_freq,
                 train_freq,
                 reward_scaling,
                 policy_freq_high,
                 policy_freq_low) -> None:
        """
        Constructor for the HIRO agent.
        """

        self.subgoal = Subgoal(subgoal_dim)
        scale_high = self.subgoal.action_space.high * np.ones(subgoal_dim)

        self.model_save_freq = model_save_freq

        self.high_con = None

        self.low_con = None

        self.low_level_replay_buffer = LowerControllerReplayBuffer(buffer_size, batch_size)
        self.high_level_replay_buffer = HigherControllerReplayBuffer(buffer_size, batch_size)

        self.buffer_freq = buffer_freq

        self.train_freq = train_freq
        self.reward_scaling = reward_scaling
        self.episode_subreward = 0
        self.sr = 0
        self.buf = [None, None, None, 0, None, None, [], []]
        self.fg = np.array([0, 0])
        self.sg = self.subgoal.action_space.sample()

        self.start_training_steps = start_training_steps

    def step(self, s, env, step, global_step=0, explore=False):
        """
        step in the environment.
        """
        # Lower Level Controller
        if explore:
            # Take random action for start_training_steps
            if global_step < self.start_training_steps:
                a = env.action_space.sample()
            else:
                a = self._choose_action_with_noise(s, self.sg)
        else:
            a = self._choose_action(s, self.sg)

        obs, r, done, _ = env.step(a)
        n_s = obs['observation']

        # Higher Level Controller
        # Take random action for start_training steps
        if explore:
            if global_step < self.start_training_steps:
                n_sg = self.subgoal.action_space.sample()
            else:
                n_sg = self._choose_subgoal_with_noise(step, s, self.sg, n_s)
        else:
            n_sg = self._choose_subgoal(step, s, self.sg, n_s)

        self.n_sg = n_sg

        return a, r, n_s, done

    def append(self, step, s, a, n_s, r, d):
        """
        add experiences to low and high level replay buffers.
        """
        self.sr = self.low_reward(s, self.sg, n_s)

        # Low Replay Buffer
        self.low_level_replay_buffer.append(s, self.sg, a, self.sr, n_s,
                                            self.n_sg, is_state_terminal=d)

        # High Replay Buffer
        if _is_update(step, self.buffer_freq, rem=1):
            if len(self.buf[6]) == self.buffer_freq:
                self.buf[4] = s
                self.buf[5] = float(d)
                self.high_level_replay_buffer.append(
                    state=self.buf[0],
                    goal=self.buf[1],
                    action=self.buf[2],
                    reward=self.buf[3],
                    n_state=self.buf[4],
                    is_state_terminal=self.buf[5],
                    state_arr=np.array(self.buf[6]),
                    action_arr=np.array(self.buf[7])
                )

            self.buf = [s, self.fg, self.sg, 0, None, None, [], []]

        self.buf[3] += self.reward_scaling * r
        self.buf[6].append(s)
        self.buf[7].append(a)

    def save(self, episode):
        self.low_con.save(episode)
        self.high_con.save(episode)

    def load(self, episode):
        self.low_con.load(episode)
        self.high_con.load(episode)
