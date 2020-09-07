import torch
import numpy as np
import os

from pfrl.agent import HRLAgent
from pfrl.replay_buffers import (
    LowerControllerReplayBuffer,
    HigherControllerReplayBuffer
)
from pfrl.agents.hrl.hrl_controllers import (
    LowerController,
    HigherController
)
from pfrl.utils import _is_update, _mean_or_nan


class HIROAgent(HRLAgent):
    def __init__(self,
                 state_dim,
                 action_dim,
                 goal_dim,
                 subgoal_dim,
                 subgoal_space,
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
                 policy_freq_low,
                 gpu) -> None:
        """
        Constructor for the HIRO agent.
        """
        # get scale for subgoal
        self.scale_high = subgoal_space.high * np.ones(subgoal_dim)
        self.scale_low = scale_low
        self.model_save_freq = model_save_freq

        # create replay buffers
        low_level_replay_buffer = LowerControllerReplayBuffer(buffer_size)
        high_level_replay_buffer = HigherControllerReplayBuffer(buffer_size)

        # higher td3 controller
        self.high_con = HigherController(
            state_dim=state_dim,
            goal_dim=goal_dim,
            action_dim=subgoal_dim,
            scale=self.scale_high,
            model_path=model_path,
            policy_freq=policy_freq_high,
            replay_buffer=high_level_replay_buffer,
            minibatch_size=batch_size,
            gpu=gpu
        )

        # lower td3 controller
        self.low_con = LowerController(
            state_dim=state_dim,
            goal_dim=subgoal_dim,
            action_dim=action_dim,
            scale=self.scale_low,
            model_path=model_path,
            policy_freq=policy_freq_low,
            replay_buffer=low_level_replay_buffer,
            minibatch_size=batch_size,
            gpu=gpu
        )

        self.buffer_freq = buffer_freq

        self.train_freq = train_freq
        self.reward_scaling = reward_scaling
        self.episode_subreward = 0
        self.sr = 0
        self.state_arr = []
        self.action_arr = []
        self.cumulative_reward = 0
        self.last_high_level_obs = None
        self.last_high_level_goal = None
        self.last_high_level_action = None

        self.start_training_steps = start_training_steps

    def act_high_level(self, obs, goal, subgoal, step=0):
        """
        high level actor
        """
        n_sg = self._choose_subgoal(step, self.last_obs, subgoal, obs, goal)
        self.sr = self.low_reward(self.last_obs, subgoal, obs)
        # clip values
        n_sg = np.clip(n_sg, a_min=-self.scale_high, a_max=self.scale_high)
        return n_sg

    def act_low_level(self, obs, goal):
        """
        low level actor,
        conditioned on an observation and goal.
        """
        self.last_obs = obs
        # goal = self.sg
        self.last_action = self.low_con.policy(obs, goal)
        self.last_action = np.clip(self.last_action, a_min=-self.scale_low, a_max=self.scale_low)
        return self.last_action

    def observe(self, obs, goal, subgoal, reward, done, reset, global_step=0, start_training_steps=0):
        """
        after getting feedback from the environment, observe,
        and train both the low and high level controllers.
        """

        if global_step >= start_training_steps:
            # start training once the global step surpasses
            # the start training steps
            self.low_con.observe(obs, subgoal, self.sr, done)

            if global_step % self.train_freq == 0 and len(self.action_arr) == self.train_freq:
                # train high level controller every self.train_freq steps
                self.high_con.agent.update_high_level_last_results(self.last_high_level_obs, self.last_high_level_goal, self.last_high_level_action)
                self.high_con.observe(self.low_con, self.state_arr, self.action_arr, self.cumulative_reward, goal, obs, done)
                self.action_arr = []
                self.state_arr = []

                # reset last high level obs, goal, action
                self.last_high_level_obs = torch.FloatTensor(obs)
                self.last_high_level_goal = torch.FloatTensor(goal)
                self.last_high_level_action = subgoal
                self.cumulative_reward = 0

            elif global_step % self.train_freq == 0 and self.last_high_level_obs is None:
                self.last_high_level_obs = torch.FloatTensor(obs)
                self.last_high_level_goal = torch.FloatTensor(goal)
                self.last_high_level_action = subgoal

            self.action_arr.append(self.last_action)
            self.state_arr.append(self.last_obs)
            self.cumulative_reward += (self.reward_scaling * reward)

    def _choose_subgoal(self, step, s, sg, n_s, goal):
        """
        chooses the next subgoal for the low level controller.
        """
        if step % self.buffer_freq == 0:
            sg = self.high_con.policy(s, goal)
        else:
            sg = self.subgoal_transition(s, sg, n_s)

        return sg

    def subgoal_transition(self, s, sg, n_s):
        """
        subgoal transition function, provided as input to the low
        level controller.
        """
        return s[:sg.shape[0]] + sg - n_s[:sg.shape[0]]

    def low_reward(self, s, sg, n_s):
        """
        reward function for low level controller.
        """
        abs_s = s[:sg.shape[0]] + sg
        return -np.sqrt(np.sum((abs_s - n_s[:sg.shape[0]])**2))

    def end_step(self):
        """
        ends a step within an episode.
        """
        self.episode_subreward += self.sr
        self.sg = self.n_sg

    def end_episode(self, episode, logger=None):
        """
        ends a full episode.
        """
        if logger:
            # log
            logger.write('reward/Intrinsic Reward', self.episode_subreward, episode)

            # Save Model
            if _is_update(episode, self.model_save_freq):
                self.save(episode=episode)

        self.episode_subreward = 0
        self.sr = 0

    def save(self, episode):
        """
        saves the model, aka the lower and higher controllers' parameters.
        """
        low_controller_dir = f'models/low_controller/episode_{episode}'
        high_controller_dir = f'models/high_controller/episode_{episode}'

        os.makedirs(low_controller_dir, exist_ok=True)
        os.makedirs(high_controller_dir, exist_ok=True)

        self.low_con.save(low_controller_dir)
        self.high_con.save(high_controller_dir)

    def load(self, episode):
        """
        loads from an episode.
        """
        low_controller_dir = f'models/low_controller/episode_{episode}'
        high_controller_dir = f'models/high_controller/episode_{episode}'
        try:
            self.low_con.load(low_controller_dir)
            self.high_con.load(high_controller_dir)
        except Exception as e:
            raise NotADirectoryError("Directory for loading internal state not found!")

    def change_to_eval(self):
        """
        sets an agent to eval - making
        for the deterministic policy of td3
        """
        self.training = False
        self.low_con.agent.training = False
        self.high_con.agent.training = False

    def set_to_train_(self):
        """
        sets an agent to train - this
        will make for a non-deterministic policy.
        """
        self.low_con.agent.training = True
        self.high_con.agent.training = True

    def set_to_eval_(self):
        """
        sets an agent to eval - making
        for the deterministic policy of td3
        """
        self.low_con.agent.training = False
        self.high_con.agent.training = False

    def get_statistics(self):
        """
        gets the statistics of all of the actors and critics for the high
        and low level controllers in the HIRO algorithm.
        """
        return [
            ("low_con_average_q1", _mean_or_nan(self.low_con.agent.q1_record)),
            ("low_con_average_q2", _mean_or_nan(self.low_con.agent.q2_record)),
            ("low_con_average_q_func1_loss", _mean_or_nan(self.low_con.agent.q_func1_loss_record)),
            ("low_con_average_q_func2_loss", _mean_or_nan(self.low_con.agent.q_func2_loss_record)),
            ("low_con_average_policy_loss", _mean_or_nan(self.low_con.agent.policy_loss_record)),
            ("low_con_policy_n_updates", self.low_con.agent.policy_n_updates),
            ("low_con_q_func_n_updates", self.low_con.agent.q_func_n_updates),

            ("high_con_average_q1", _mean_or_nan(self.high_con.agent.q1_record)),
            ("high_con_average_q2", _mean_or_nan(self.high_con.agent.q2_record)),
            ("high_con_average_q_func1_loss", _mean_or_nan(self.high_con.agent.q_func1_loss_record)),
            ("high_con_average_q_func2_loss", _mean_or_nan(self.high_con.agent.q_func2_loss_record)),
            ("high_con_average_policy_loss", _mean_or_nan(self.high_con.agent.policy_loss_record)),
            ("high_con_policy_n_updates", self.high_con.agent.policy_n_updates),
            ("high_con_q_func_n_updates", self.high_con.agent.q_func_n_updates),
        ]
