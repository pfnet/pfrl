import os

import numpy as np
import torch

from pfrl.agent import HRLAgent
from pfrl.replay_buffers import (
    LowerControllerReplayBuffer,
    HigherControllerReplayBuffer
)
from pfrl.agents.hrl.hrl_controllers import (
    LowerController,
    HigherController
)
from pfrl.utils import _mean_or_nan


class HIROAgent(HRLAgent):
    def __init__(self,
                 state_dim,
                 action_dim,
                 goal_dim,
                 subgoal_dim,
                 high_level_burnin_action_func,
                 low_level_burnin_action_func,
                 scale_low,
                 scale_high,
                 model_path,
                 buffer_size,
                 batch_size,
                 subgoal_freq,
                 train_freq,
                 reward_scaling,
                 policy_freq_high,
                 policy_freq_low,
                 gpu,
                 start_training_steps=2500):
        """
        Constructor for the HIRO agent.
        """
        # get scale for subgoal
        self.scale_high = scale_high
        self.scale_low = scale_low

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
            gpu=gpu,
            burnin_action_func=high_level_burnin_action_func
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
            gpu=gpu,
            burnin_action_func=low_level_burnin_action_func
        )

        self.subgoal_freq = subgoal_freq

        self.train_freq = train_freq
        self.reward_scaling = reward_scaling
        self.start_training_steps = start_training_steps
        self.sr = 0
        self.state_arr = []
        self.action_arr = []
        self.cumulative_reward = 0
        self.last_high_level_obs = None
        self.last_high_level_goal = None
        self.last_high_level_action = None
        self.last_subgoal = None

    def act_high_level(self, obs, goal, subgoal, step=0, global_step=0):
        """
        high level actor
        """
        self.last_subgoal = subgoal
        if global_step < self.start_training_steps and self.training == True:
            n_sg = self.high_con.policy(self.last_obs, goal)
        else:
            n_sg = self._choose_subgoal(step, self.last_obs, subgoal, obs, goal)
        self.sr = self._low_reward(self.last_obs, subgoal, obs)
        # clip values
        return n_sg

    def act_low_level(self, obs, goal):
        """
        low level actor,
        conditioned on an observation and goal.
        """
        self.last_obs = obs

        self.last_action = self.low_con.policy(obs, goal)
        return self.last_action

    def observe(self, obs, goal, subgoal, reward, done, reset, step=0):
        """
        after getting feedback from the environment, observe,
        and train both the low and high level controllers.
        """
        if self.training:
            # start training once the global step surpasses
            # the start training steps
            self.low_con.observe(obs, subgoal, self.sr, done)
            if step != 0 and step % self.train_freq == 1:
                if len(self.action_arr) == self.train_freq:
                    # train high level controller every self.train_freq steps
                    self.high_con.agent.update_high_level_last_results(self.last_high_level_obs, self.last_high_level_goal, self.last_high_level_action)
                    self.high_con.observe(self.low_con, self.state_arr, self.action_arr, self.cumulative_reward, goal, obs, done)

                # reset last high level obs, goal, action
                self.action_arr = []
                self.state_arr = []
                self.last_high_level_obs = torch.FloatTensor(self.last_obs)
                self.last_high_level_goal = torch.FloatTensor(goal)
                self.last_high_level_action = self.last_subgoal
                self.cumulative_reward = 0

            self.action_arr.append(self.last_action)
            self.state_arr.append(self.last_obs)
            self.cumulative_reward += (self.reward_scaling * reward)

    def end_episode(self):
        self.action_arr = []
        self.state_arr = []
        self.last_high_level_obs = None
        self.last_high_level_goal = None
        self.last_high_level_action = None
        self.cumulative_reward = 0

    def _choose_subgoal(self, step, s, sg, n_s, goal):
        """
        chooses the next subgoal for the low level controller.
        """
        if step % self.subgoal_freq == 0:
            sg = self.high_con.policy(s, goal)
        else:
            sg = self._subgoal_transition(s, sg, n_s)

        return sg

    def _subgoal_transition(self, s, sg, n_s):
        """
        subgoal transition function, provided as input to the low
        level controller.
        """
        return s[:sg.shape[0]] + sg - n_s[:sg.shape[0]]

    def _low_reward(self, s, sg, n_s):
        """
        reward function for low level controller.
        """
        abs_s = s[:sg.shape[0]] + sg
        return -np.sqrt(np.sum((abs_s - n_s[:sg.shape[0]])**2))

    def save(self, outdir):
        """
        saves the model, aka the lower and higher controllers' parameters.
        """
        low_controller_dir = os.path.join(outdir, 'low_controller')
        high_controller_dir = os.path.join(outdir, 'high_controller')

        os.makedirs(low_controller_dir, exist_ok=True)
        os.makedirs(high_controller_dir, exist_ok=True)

        self.low_con.save(low_controller_dir)
        self.high_con.save(high_controller_dir)

    def load(self, outdir):
        """
        loads from an episode.
        """
        low_controller_dir = os.path.join(outdir, 'low_controller')
        high_controller_dir = os.path.join(outdir, 'high_controller')
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
        print("evaluation mode turned on")
        self.training = False
        self.low_con.agent.training = False
        self.high_con.agent.training = False

    def change_to_train(self):
        """
        sets an agent to train - including
        some exploration
        """
        print("training mode turned on")
        self.training = True
        self.low_con.agent.training = True
        self.high_con.agent.training = True

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
