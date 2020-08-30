import argparse
import functools
import os

import gym
import gym.spaces
import numpy as np
import torch
from torch import nn
from pybullet_robot_envs.envs.panda_envs.panda_env import pandaEnv

import pfrl
from pfrl import utils
from pfrl import experiments
from pfrl.agents.hrl.hiro_agent import HIROAgent


class CastAction(gym.ActionWrapper):
    """Cast actions to a given type."""

    def __init__(self, env, type_):
        super().__init__(env)
        self.type_ = type_

    def action(self, action):
        return self.type_(action)


class TransposeObservation(gym.ObservationWrapper):
    """Transpose observations."""

    def __init__(self, env, axes):
        super().__init__(env)
        self._axes = axes
        assert isinstance(env.observation_space, gym.spaces.Box)
        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low.transpose(*self._axes),
            high=env.observation_space.high.transpose(*self._axes),
            dtype=env.observation_space.dtype,
        )

    def observation(self, observation):
        return observation.transpose(*self._axes)


class ObserveElapsedSteps(gym.Wrapper):
    """Observe the number of elapsed steps in an episode.

    A new observation will be a tuple of an original observation and an integer
    that is equal to the elapsed steps in an episode.
    """

    def __init__(self, env, max_steps):
        super().__init__(env)
        self._max_steps = max_steps
        self._elapsed_steps = 0
        self.observation_space = gym.spaces.Tuple(
            (env.observation_space, gym.spaces.Discrete(self._max_steps + 1),)
        )

    def reset(self):
        self._elapsed_steps = 0
        return self.env.reset(), self._elapsed_steps

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        assert self._elapsed_steps <= self._max_steps
        return (observation, self._elapsed_steps), reward, done, info


class RecordMovie(gym.Wrapper):
    """Record MP4 videos using pybullet's logging API."""

    def __init__(self, env, dirname):
        super().__init__(env)
        self._episode_idx = -1
        self._dirname = dirname

    def reset(self):
        obs = self.env.reset()
        self._episode_idx += 1
        import pybullet

        pybullet.startStateLogging(
            pybullet.STATE_LOGGING_VIDEO_MP4,
            os.path.join(self._dirname, "{}.mp4".format(self._episode_idx)),
        )
        return obs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help=(
            "Directory path to save output files."
            " If it does not exist, it will be created."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 31)")
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU to use, set to -1 if no GPU."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        default=False,
        help="Evaluate the agent without training.",
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Load a saved agent from a given directory.",
    )
    parser.add_argument(
        "--final-exploration-steps",
        type=int,
        default=5 * 10 ** 5,
        help="Timesteps after which we stop annealing exploration rate",
    )
    parser.add_argument(
        "--final-epsilon",
        type=float,
        default=0.2,
        help="Final value of epsilon during training.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2 * 10 ** 6,
        help="Total number of timesteps to train the agent.",
    )
    parser.add_argument(
        "--replay-start-size",
        type=int,
        default=5 * 10 ** 4,
        help="Minimum replay buffer size before performing gradient updates.",
    )
    parser.add_argument(
        "--target-update-interval",
        type=int,
        default=1 * 10 ** 4,
        help="Frequency (in timesteps) at which the target network is updated.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=10 ** 5,
        help="Frequency (in timesteps) of evaluation phase.",
    )
    parser.add_argument(
        "--update-interval",
        type=int,
        default=1,
        help="Frequency (in timesteps) of network updates.",
    )
    parser.add_argument(
        "--eval-n-runs",
        type=int,
        default=100,
        help="Number of episodes used for evaluation.",
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level. 10:DEBUG, 20:INFO etc.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Render env states in a GUI window.",
    )
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument(
        "--num-envs", type=int, default=1, help="Number of envs run in parallel."
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size used for training."
    )
    parser.add_argument(
        "--record",
        action="store_true",
        default=False,
        help="Record videos of evaluation envs. --render should also be specified.",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    args = parser.parse_args()

    import logging

    logging.basicConfig(level=args.log_level)

    # Set a random seed used in PFRL.
    utils.set_random_seed(args.seed)

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2 ** 32

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    print("Output files are saved in {}".format(args.outdir))

    max_episode_steps = 8

    def make_panda_env(idx, test):
        from pybullet_robot_envs.envs.panda_envs.panda_push_gym_goal_env import (
            pandaPushGymGoalEnv
        )  # NOQA

        # use different seeds for train vs test envs
        process_seed = int(process_seeds[idx])
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        utils.set_random_seed(env_seed)
        env = pandaPushGymGoalEnv(renders=args.render and (args.demo or not test),
                                  max_steps=max_episode_steps)

        env.seed(int(env_seed))

        if test and args.record:
            assert args.render, "To use --record, --render needs be specified."
            video_dir = os.path.join(args.outdir, "video_{}".format(idx))
            os.mkdir(video_dir)
            env = RecordMovie(env, video_dir)
        return env

    def make_batch_panda_env(test):
        return pfrl.envs.MultiprocessVectorEnv(
            [functools.partial(make_panda_env, idx, test) for idx in range(args.num_envs)]
        )

    # eval_env = make_batch_panda_env(test=True)
    eval_env = make_panda_env(0, test=True)

    env_state_dim = eval_env.observation_space.spaces['observation'].shape[0]
    env_action_dim = eval_env.action_space.shape[0]

    env_goal_dim = 5
    gpu = 0 if torch.cuda.is_available() else None
    agent = HIROAgent(state_dim=env_state_dim,
                      action_dim=env_action_dim,
                      goal_dim=env_goal_dim,
                      subgoal_dim=7,
                      scale_low=1,
                      start_training_steps=100,
                      model_save_freq=10,
                      model_path='model',
                      buffer_size=200000,
                      batch_size=100,
                      buffer_freq=10,
                      train_freq=10,
                      reward_scaling=0.1,
                      policy_freq_high=2,
                      policy_freq_low=2,
                      gpu=gpu)

    if args.load:
        # load weights from agent if arg supplied
        agent.load(args.load)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=eval_env, agent=agent, n_steps=None, n_episodes=args.eval_n_runs
        )
        print(
            "n_runs: {} mean: {} median: {} stdev {}".format(
                args.eval_n_runs,
                eval_stats["mean"],
                eval_stats["median"],
                eval_stats["stdev"],
            )
        )
    else:
        experiments.train_hrl_agent_with_evaluation(
            agent=agent,
            env=make_panda_env(0, test=False),
            eval_env=make_panda_env(0, test=True),
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            outdir=args.outdir,
            save_best_so_far_agent=False,
        )


if __name__ == "__main__":
    main()
