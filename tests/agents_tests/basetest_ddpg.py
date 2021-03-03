import numpy as np
import torch
from basetest_training import _TestTraining
from torch import nn

from pfrl import replay_buffers
from pfrl.envs.abc import ABC
from pfrl.explorers.epsilon_greedy import LinearDecayEpsilonGreedy
from pfrl.nn import BoundByTanh, ConcatObsAndAction, RecurrentSequential
from pfrl.policies import DeterministicHead


class _TestDDPGOnABC(_TestTraining):
    def make_agent(self, env, gpu):
        policy, q_func = self.make_model(env)

        actor_opt = torch.optim.Adam(policy.parameters(), lr=1e-4)
        critic_opt = torch.optim.Adam(q_func.parameters(), lr=1e-3)

        explorer = self.make_explorer(env)
        rbuf = self.make_replay_buffer(env)
        return self.make_ddpg_agent(
            env=env,
            policy=policy,
            q_func=q_func,
            actor_opt=actor_opt,
            critic_opt=critic_opt,
            explorer=explorer,
            rbuf=rbuf,
            gpu=gpu,
        )

    def make_ddpg_agent(
        self,
        env,
        policy,
        q_func,
        actor_opt,
        critic_opt,
        explorer,
        rbuf,
        gpu,
    ):
        raise NotImplementedError()

    def make_explorer(self, env):
        def random_action_func():
            a = env.action_space.sample()
            if isinstance(a, np.ndarray):
                return a.astype(np.float32)
            else:
                return a

        return LinearDecayEpsilonGreedy(1.0, 0.2, 1000, random_action_func)

    def make_replay_buffer(self, env):
        return replay_buffers.ReplayBuffer(10 ** 5)


class _TestDDPGOnContinuousPOABC(_TestDDPGOnABC):
    def make_model(self, env):
        obs_size = env.observation_space.low.size
        action_size = env.action_space.low.size
        hidden_size = 50
        # Model must be recurrent
        policy = RecurrentSequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.LSTM(input_size=hidden_size, hidden_size=hidden_size),
            nn.Linear(hidden_size, action_size),
            BoundByTanh(low=env.action_space.low, high=env.action_space.high),
            DeterministicHead(),
        )
        q_func = RecurrentSequential(
            ConcatObsAndAction(),
            nn.Linear(obs_size + action_size, hidden_size),
            nn.ReLU(),
            nn.LSTM(input_size=hidden_size, hidden_size=hidden_size),
            nn.Linear(hidden_size, 1),
        )
        return policy, q_func

    def make_env_and_successful_return(self, test):
        return ABC(discrete=False, partially_observable=True, deterministic=test), 1

    def make_replay_buffer(self, env):
        return replay_buffers.EpisodicReplayBuffer(10 ** 5)


class _TestDDPGOnContinuousABC(_TestDDPGOnABC):
    def make_model(self, env):
        obs_size = env.observation_space.low.size
        action_size = env.action_space.low.size
        hidden_size = 50
        policy = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            BoundByTanh(low=env.action_space.low, high=env.action_space.high),
            DeterministicHead(),
        )
        q_func = nn.Sequential(
            ConcatObsAndAction(),
            nn.Linear(obs_size + action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        return policy, q_func

    def make_env_and_successful_return(self, test):
        return ABC(discrete=False, deterministic=test), 1
