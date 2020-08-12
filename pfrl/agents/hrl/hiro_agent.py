from typing import Any
import torch
import numpy as np
from pfrl.agent import HRLAgent, Agent
from pfrl.utils import Subgoal
from pfrl.replay_buffers import (
    LowerControllerReplayBuffer,
    HigherControllerReplayBuffer
)
from pfrl.agents import TD3


def _is_update(episode, freq, ignore=0, rem=0):
    if episode != ignore and episode % freq == rem:
        return True
    return False

# standard controller


class HRLControllerBase():
    def __init__(
            self,
            agent: Agent,
            state_dim,
            goal_dim,
            action_dim,
            scale,
            model_path,
            name,
            actor_lr=0.0001,
            critic_lr=0.001,
            expl_noise=0.1,
            policy_noise=0.2,
            noise_clip=0.5,
            gamma=0.99,
            policy_freq=2,
            tau=0.005):
        # example name- 'td3_low' or 'td3_high'
        self.name = name
        self.scale = scale
        self.model_path = model_path
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # parameters
        self.expl_noise = expl_noise
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.gamma = gamma
        self.policy_freq = policy_freq
        self.tau = tau
        # create td3 agent
        self.agent = TD3()

        self.actor = TD3Actor(state_dim, goal_dim, action_dim, scale=scale).to(device)
        self.actor_target = TD3Actor(state_dim, goal_dim, action_dim, scale=scale).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic1 = TD3Critic(state_dim, goal_dim, action_dim).to(device)
        self.critic2 = TD3Critic(state_dim, goal_dim, action_dim).to(device)
        self.critic1_target = TD3Critic(state_dim, goal_dim, action_dim).to(device)
        self.critic2_target = TD3Critic(state_dim, goal_dim, action_dim).to(device)

        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)
        self._initialize_target_networks()

        self._initialized = False
        self.total_it = 0

    def save(self):
        """
        save the internal state of the TD3 agent.
        """
        self.agent.save('dirname')

    def load(self):
        """
        load the internal state of the TD3 agent.
        """
        self.agent.load('dirname')

    def policy(self, state, goal):
        return self.agent.act([torch.concat(state, goal)])

    def _train(self, states, goals, actions, rewards, n_states, n_goals, not_done):
        self.agent.batch_observe([states, goals], rewards, not_done, None)

    def train(self, replay_buffer, iterations=1):
        states, goals, actions, n_states, rewards, not_done = replay_buffer.sample()
        return self._train(states, goals, actions, rewards, n_states, goals, not_done)

    def policy_with_noise(self, state, goal, to_numpy=True):
        action = self.policy(state, goal)
        action += self._sample_exploration_noise(action)
        action = torch.min(action,  self.actor.scale)
        action = torch.max(action, -self.actor.scale)

        return action.squeeze()

    def _sample_exploration_noise(self, actions):
        mean = torch.zeros(actions.size()).to(self.device)
        var = torch.ones(actions.size()).to(self.device)
        # expl_noise = self.expl_noise - (self.expl_noise/1200) * (self.total_it//10000)
        return torch.normal(mean, self.expl_noise*var)


# lower controller
class LowerController(HRLControllerBase):
    pass


class HigherController(HRLControllerBase):
    pass
# higher controller


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

        self.high_con = HigherController(
            state_dim=state_dim,
            goal_dim=goal_dim,
            action_dim=subgoal_dim,
            scale=scale_high,
            model_path=model_path,
            policy_freq=policy_freq_high
            )

        self.low_con = LowerController(
            state_dim=state_dim,
            goal_dim=subgoal_dim,
            action_dim=action_dim,
            scale=scale_low,
            model_path=model_path,
            policy_freq=policy_freq_low
            )

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

    def train(self, global_step) -> Any:
        losses = {}
        td_errors = {}

        if global_step >= self.start_training_steps:
            # start training once the global step surpasses
            # the start training steps
            loss, td_error = self.low_con.train(self.low_level_replay_buffer)
            # update losses
            losses.update(loss)
            td_errors.update(td_error)

            if global_step % self.train_freq == 0:
                # train high level controller every self.train_freq steps
                loss, td_error = self.high_con.train(self.high_level_replay_buffer, self.low_con)
                losses.update(loss)
                td_errors.update(td_error)
        return losses, td_errors

    def _choose_action_with_noise(self, s, sg):
        """
        selects an action.
        """
        pass

    def _choose_subgoal_with_noise(self, step, s, sg, n_s):
        """
        selects a subgoal for the low level controller, with noise.
        """
        pass

    def _choose_action(self, s, sg):
        """
        runs the policy of the low level controller.
        """
        pass

    def _choose_subgoal(self, step, s, sg, n_s):
        """
        chooses the next subgoal for the low level controller.
        """
        pass

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
        self.episode_subreward += self.sr
        self.sg = self.n_sg

    def end_episode(self, episode, logger=None):
        if logger:
            # log
            logger.write('reward/Intrinsic Reward', self.episode_subreward, episode)

            # Save Model
            if _is_update(episode, self.model_save_freq):
                self.save(episode=episode)

        self.episode_subreward = 0
        self.sr = 0
        self.buf = [None, None, None, 0, None, None, [], []]

    def save(self, episode):
        self.low_con.save(episode)
        self.high_con.save(episode)

    def load(self, episode):
        self.low_con.load(episode)
        self.high_con.load(episode)
