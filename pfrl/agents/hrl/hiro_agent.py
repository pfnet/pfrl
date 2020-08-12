from typing import Any
import torch
from torch import nn
import numpy as np

import pfrl
from pfrl.agent import HRLAgent
from pfrl.utils import Subgoal
from pfrl.replay_buffers import (
    LowerControllerReplayBuffer,
    HigherControllerReplayBuffer
)
from pfrl import explorers, replay_buffers

from pfrl.agents import TD3


def _is_update(episode, freq, ignore=0, rem=0):
    if episode != ignore and episode % freq == rem:
        return True
    return False


def var(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor


def get_tensor(z):
    if len(z.shape) == 1:
        return var(torch.FloatTensor(z.copy())).unsqueeze(0)
    else:
        return var(torch.FloatTensor(z.copy()))

# standard controller


class HRLControllerBase():
    def __init__(
            self,
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

        policy = nn.Sequential(
            nn.Linear(state_dim + goal_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, action_dim),
            nn.Tanh(),
            pfrl.policies.DeterministicHead(),
            )
        policy_optimizer = torch.optim.Adam(policy.parameters(), lr=actor_lr)

        def make_q_func_with_optimizer():
            q_func = nn.Sequential(
                pfrl.nn.ConcatObsAndAction(),
                nn.Linear(state_dim + goal_dim + action_dim, 400),
                nn.ReLU(),
                nn.Linear(400, 300),
                nn.ReLU(),
                nn.Linear(300, 1),
            )
            q_func_optimizer = torch.optim.Adam(q_func.parameters(), lr=critic_lr)
            return q_func, q_func_optimizer

        q_func1, q_func1_optimizer = make_q_func_with_optimizer()
        q_func2, q_func2_optimizer = make_q_func_with_optimizer()

        rbuf = replay_buffers.ReplayBuffer(10 ** 6)
        explorer = explorers.AdditiveGaussian(
            scale=0.1, low=-1, high=1
        )

        def burnin_action_func():
            """
            Select random actions until model is updated one or more times.
            """
            return np.random.uniform(-1, 1).astype(np.float32)
        """
        agent = pfrl.agents.TD3(
        policy,
        q_func1,
        q_func2,
        policy_optimizer,
        q_func1_optimizer,
        q_func2_optimizer,
        rbuf,
        gamma=0.99,
        soft_update_tau=5e-3,
        explorer=explorer,
        replay_start_size=args.replay_start_size,
        gpu=args.gpu,
        minibatch_size=args.batch_size,
        burnin_action_func=burnin_action_func,
    )
        """
        self.agent = TD3(
            policy,
            q_func1,
            q_func2,
            policy_optimizer,
            q_func1_optimizer,
            q_func2_optimizer,
            rbuf,
            gamma=gamma,
            soft_update_tau=tau,
            explorer=explorer,
            update_interval=policy_freq,
            burnin_action_func=burnin_action_func
        )

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
        """
        run the policy (actor).
        """
        return self.agent.act([torch.cat([state, goal])])

    def _train(self, states, goals, actions, rewards, n_states, n_goals, not_done):
        """
        train the model.
        """
        self.agent.batch_observe(torch.cat([states, goals]), rewards, not_done, None)

    def train(self, replay_buffer, iterations=1):
        """
        get data from the replay buffer, and train.
        """
        states, goals, actions, n_states, rewards, not_done = replay_buffer.sample()
        return self._train(states, goals, actions, rewards, n_states, goals, not_done)

    def policy_with_noise(self, state, goal):
        """
        run the policy...with a little extra noise.
        """
        action = self.policy(state, goal)
        action += self._sample_exploration_noise(action)
        action = torch.min(action,  self.actor.scale)
        action = torch.max(action, -self.actor.scale)
        # to-do - is this needed?
        return action.squeeze()

    def _sample_exploration_noise(self, actions):
        """
        add a bit of noise to the policy to encourage exploration.
        """
        mean = torch.zeros(actions.size()).to(self.device)
        var = torch.ones(actions.size()).to(self.device)
        # expl_noise = self.expl_noise - (self.expl_noise/1200) * (self.total_it//10000)
        return torch.normal(mean, self.expl_noise*var)


# lower controller
class LowerController(HRLControllerBase):
    def __init__(
            self,
            state_dim,
            goal_dim,
            action_dim,
            scale,
            model_path,
            name,
            actor_lr=0.0001,
            critic_lr=0.001,
            expl_noise=1.0,
            policy_noise=0.2,
            noise_clip=0.5,
            gamma=0.99,
            policy_freq=2,
            tau=0.005):
        super(LowerController, self).__init__(
                                            state_dim,
                                            goal_dim,
                                            action_dim,
                                            scale,
                                            model_path,
                                            name,
                                            actor_lr,
                                            critic_lr,
                                            expl_noise,
                                            policy_noise,
                                            noise_clip,
                                            gamma,
                                            policy_freq,
                                            tau)
        self.name = name

    def train(self, replay_buffer):
        if not self._initialized:
            self.agent.sync_target_network()

        states, sgoals, actions, n_states, n_sgoals, rewards, not_done = replay_buffer.sample()

        return self._train(states, sgoals, actions, rewards, n_states, n_sgoals, not_done)


# higher controller

class HigherController(HRLControllerBase):
    def __init__(
            self,
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
        super(HigherController, self).__init__(
                                                state_dim,
                                                goal_dim,
                                                action_dim,
                                                scale,
                                                model_path,
                                                name,
                                                actor_lr,
                                                critic_lr,
                                                expl_noise,
                                                policy_noise,
                                                noise_clip,
                                                gamma,
                                                policy_freq,
                                                tau)
        self.name = 'high'
        self.action_dim = action_dim

    def off_policy_corrections(self, low_con, batch_size, sgoals, states, actions, candidate_goals=8):
        """
        implementation of off policy correction in HIRO paper.
        """

        first_s = [s[0] for s in states]  # First x
        last_s = [s[-1] for s in states]  # Last x

        # Shape: (batch_size, 1, subgoal_dim)
        # diff = 1
        # different in goals
        diff_goal = (np.array(last_s) -
                     np.array(first_s))[:, np.newaxis, :self.action_dim]

        # Shape: (batch_size, 1, subgoal_dim)
        # original = 1
        # random = candidate_goals
        original_goal = np.array(sgoals)[:, np.newaxis, :]
        # select random goals
        random_goals = np.random.normal(loc=diff_goal, scale=.5*self.scale[None, None, :],
                                        size=(batch_size, candidate_goals, original_goal.shape[-1]))
        random_goals = random_goals.clip(-self.scale, self.scale)

        # Shape: (batch_size, 10, subgoal_dim)
        candidates = np.concatenate([original_goal, diff_goal, random_goals], axis=1)
        # states = np.array(states)[:, :-1, :]
        actions = np.array(actions)
        seq_len = len(states[0])

        # For ease
        new_batch_sz = seq_len * batch_size
        action_dim = actions[0][0].shape
        obs_dim = states[0][0].shape
        ncands = candidates.shape[1]

        true_actions = actions.reshape((new_batch_sz,) + action_dim)
        observations = states.reshape((new_batch_sz,) + obs_dim)
        goal_shape = (new_batch_sz, self.action_dim)
        # observations = get_obs_tensor(observations, sg_corrections=True)

        # batched_candidates = np.tile(candidates, [seq_len, 1, 1])
        # batched_candidates = batched_candidates.transpose(1, 0, 2)

        policy_actions = np.zeros((ncands, new_batch_sz) + action_dim)

        for c in range(ncands):
            subgoal = candidates[:,c]
            candidate = (subgoal + states[:, 0, :self.action_dim])[:, None] - states[:, :, :self.action_dim]
            candidate = candidate.reshape(*goal_shape)
            policy_actions[c] = low_con.policy(observations, candidate)

        difference = (policy_actions - true_actions)
        difference = np.where(difference != -np.inf, difference, 0)
        difference = difference.reshape((ncands, batch_size, seq_len) + action_dim).transpose(1, 0, 2, 3)

        logprob = -0.5*np.sum(np.linalg.norm(difference, axis=-1)**2, axis=-1)
        max_indices = np.argmax(logprob, axis=-1)
        # return best candidates with maximum probability
        return candidates[np.arange(batch_size), max_indices]

    def train(self, replay_buffer, low_con):
        """
        train the high level controller with
        the novel off policy correction.
        """
        if not self._initialized:
            self.agent.sync_target_network()
        states, goals, actions, n_states, rewards, not_done, states_arr, actions_arr = replay_buffer.sample()

        actions = self.off_policy_corrections(
            low_con,
            replay_buffer.batch_size,
            actions.cpu().data.numpy(),
            states_arr.cpu().data.numpy(),
            actions_arr.cpu().data.numpy())

        actions = get_tensor(actions)

        return self._train(states, goals, actions, rewards, n_states, goals, not_done)


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
        # create subgoal
        self.subgoal = Subgoal(subgoal_dim)
        scale_high = self.subgoal.action_space.high * np.ones(subgoal_dim)

        self.model_save_freq = model_save_freq

        # higher td3 controller
        self.high_con = HigherController(
            state_dim=state_dim,
            goal_dim=goal_dim,
            action_dim=subgoal_dim,
            scale=scale_high,
            model_path=model_path,
            policy_freq=policy_freq_high
            )

        # lower td3 controller
        self.low_con = LowerController(
            state_dim=state_dim,
            goal_dim=subgoal_dim,
            action_dim=action_dim,
            scale=scale_low,
            model_path=model_path,
            policy_freq=policy_freq_low
            )

        # create replay buffers
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
        if explore:
            # Take random action for start_training steps
            if global_step < self.start_training_steps:
                n_sg = self.subgoal.action_space.sample()
            else:
                n_sg = self._choose_subgoal_with_noise(step, s, self.sg, n_s)
        else:
            n_sg = self._choose_subgoal(step, s, self.sg, n_s)
        # next subgoal
        self.n_sg = n_sg

        return a, r, n_s, done

    def append(self, step, s, a, n_s, r, d):
        """
        add experiences to the low and high level replay buffers.
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
        if global_step >= self.start_training_steps:
            # start training once the global step surpasses
            # the start training steps
            self.low_con.train(self.low_level_replay_buffer)
            # update losses

            if global_step % self.train_freq == 0:
                # train high level controller every self.train_freq steps
                self.high_con.train(self.high_level_replay_buffer, self.low_con)

    def _choose_action_with_noise(self, s, sg):
        """
        selects an action.
        """
        return self.low_con.policy_with_noise(s, sg)

    def _choose_subgoal_with_noise(self, step, s, sg, n_s):
        """
        selects a subgoal for the low level controller, with noise.
        """
        if step % self.buffer_freq == 0:  # Should be zero
            sg = self.high_con.policy_with_noise(s, self.fg)
        else:
            sg = self.subgoal_transition(s, sg, n_s)

        return sg

    def _choose_action(self, s, sg):
        """
        runs the policy of the low level controller.
        """
        return self.low_con.policy(s, sg)

    def _choose_subgoal(self, step, s, sg, n_s):
        """
        chooses the next subgoal for the low level controller.
        """
        if step % self.buffer_freq == 0:
            sg = self.high_con.policy(s, self.fg)
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
