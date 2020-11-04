import numpy as np
import torch.nn as nn
import torch.optim as optim
from basetest_training import _TestTraining

from pfrl import q_functions, replay_buffers
from pfrl.envs.abc import ABC
from pfrl.explorers.epsilon_greedy import LinearDecayEpsilonGreedy
from pfrl.nn import RecurrentSequential


class _TestDQNLike(_TestTraining):
    def make_agent(self, env, gpu):
        q_func = self.make_q_func(env)
        opt = self.make_optimizer(env, q_func)
        explorer = self.make_explorer(env)
        rbuf = self.make_replay_buffer(env)
        agent = self.make_dqn_agent(
            env=env, q_func=q_func, opt=opt, explorer=explorer, rbuf=rbuf, gpu=gpu
        )
        return agent

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        raise NotImplementedError()

    def make_env_and_successful_return(self, test):
        raise NotImplementedError()

    def make_explorer(self, env):
        raise NotImplementedError()

    def make_optimizer(self, env, q_func):
        raise NotImplementedError()

    def make_replay_buffer(self, env):
        raise NotImplementedError()


class _TestDQNOnABC(_TestDQNLike):
    def make_agent(self, env, gpu):
        q_func = self.make_q_func(env)
        opt = self.make_optimizer(env, q_func)
        explorer = self.make_explorer(env)
        rbuf = self.make_replay_buffer(env)
        return self.make_dqn_agent(
            env=env, q_func=q_func, opt=opt, explorer=explorer, rbuf=rbuf, gpu=gpu
        )

    def make_dqn_agent(self, env, q_func, opt, explorer, rbuf, gpu):
        raise NotImplementedError()

    def make_explorer(self, env):
        def random_action_func():
            a = env.action_space.sample()
            if isinstance(a, np.ndarray):
                return a.astype(np.float32)
            else:
                return a

        return LinearDecayEpsilonGreedy(1.0, 0.5, 1000, random_action_func)

    def make_optimizer(self, env, q_func):
        opt = optim.Adam(q_func.parameters(), 1e-2)
        return opt

    def make_replay_buffer(self, env):
        return replay_buffers.ReplayBuffer(10 ** 5)


class _TestDQNOnDiscreteABC(_TestDQNOnABC):
    def make_q_func(self, env):
        return q_functions.FCStateQFunctionWithDiscreteAction(
            env.observation_space.low.size, env.action_space.n, 10, 10
        )

    def make_env_and_successful_return(self, test):
        return ABC(discrete=True, deterministic=test), 1


class _TestDQNOnDiscretePOABC(_TestDQNOnABC):
    def make_q_func(self, env):
        n_hidden_channels = 10
        return RecurrentSequential(
            nn.Linear(env.observation_space.low.size, n_hidden_channels),
            nn.ELU(),
            nn.RNN(input_size=n_hidden_channels, hidden_size=n_hidden_channels),
            nn.Linear(n_hidden_channels, env.action_space.n),
            q_functions.DiscreteActionValueHead(),
        )

    def make_replay_buffer(self, env):
        return replay_buffers.EpisodicReplayBuffer(10 ** 5)

    def make_env_and_successful_return(self, test):
        return ABC(discrete=True, partially_observable=True, deterministic=test), 1

    def make_optimizer(self, env, q_func):
        # Stabilize training by large eps
        opt = optim.Adam(q_func.parameters(), 1e-2, eps=1)
        return opt


class _TestNStepDQNOnABC(_TestDQNOnABC):
    def make_replay_buffer(self, env):
        return replay_buffers.ReplayBuffer(10 ** 5, num_steps=3)


class _TestNStepDQNOnDiscreteABC(_TestNStepDQNOnABC):
    def make_q_func(self, env):
        return q_functions.FCStateQFunctionWithDiscreteAction(
            env.observation_space.low.size, env.action_space.n, 10, 10
        )

    def make_env_and_successful_return(self, test):
        return ABC(discrete=True, deterministic=test), 1


class _TestDQNOnContinuousABC(_TestDQNOnABC):
    def make_q_func(self, env):
        n_dim_action = env.action_space.low.size
        n_dim_obs = env.observation_space.low.size
        return q_functions.FCQuadraticStateQFunction(
            n_input_channels=n_dim_obs,
            n_dim_action=n_dim_action,
            n_hidden_channels=20,
            n_hidden_layers=2,
            action_space=env.action_space,
        )

    def make_env_and_successful_return(self, test):
        return ABC(discrete=False, deterministic=test), 1


class _TestNStepDQNOnContinuousABC(_TestNStepDQNOnABC):
    def make_q_func(self, env):
        n_dim_action = env.action_space.low.size
        n_dim_obs = env.observation_space.low.size
        return q_functions.FCQuadraticStateQFunction(
            n_input_channels=n_dim_obs,
            n_dim_action=n_dim_action,
            n_hidden_channels=20,
            n_hidden_layers=2,
            action_space=env.action_space,
        )

    def make_env_and_successful_return(self, test):
        return ABC(discrete=False, deterministic=test), 1
