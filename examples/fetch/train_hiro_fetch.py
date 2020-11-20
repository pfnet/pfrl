"""
Training:
    python examples/ant/train_hiro_fetch.py
Render Trained:
    python examples/ant/train_hiro_fetch.py --render --demo --load <dir>
Example:
    python examples/ant/train_hiro_fetch.py --render --demo --load results/6900d36edd696e65e1d2ae72dd58796a2d7c19ef-34c626fd-4418a6b0/best
"""
import argparse
import functools
import os
import logging

import gym
import gym.spaces
import numpy as np
import torch

import pfrl
from pfrl import utils
from pfrl import experiments
from pfrl.agents.hrl.hiro_agent import HIROAgent

FETCH_ENVS = ['FetchReach-v1', 'FetchSlide-v1', 'FetchPush-v1', 'FetchPickAndPlace-v1']


def parse_rl_args():
    """
    parse arguments for
    training or evaluating the hiro
    agent on the ant environment.
    """
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
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use, set to -1 if no GPU.")
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
        "--add-entropy",
        type=bool,
        default=False,
        help="Whether or not to add entropy.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=16 * 10 ** 6,
        help="Total number of timesteps to train the agent.",
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
    """
    possible envs for fetch robot:

    FetchReach-v1
    FetchSlide-v1
    FetchPush-v1
    FetchPickAndPlace-v1

    """
    parser.add_argument(
        "--env",
        type=str,
        default="FetchReach-v1",
        help="OpenAI Gym Fetch robotic env env to perform algorithm on.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Render env states in a GUI window.",
    )
    parser.add_argument("--num-envs", type=int, default=1, help="Number of envs run in parallel.")
    parser.add_argument(
        "--record",
        action="store_true",
        default=False,
        help="Record videos of evaluation envs. --render should also be specified.",
    )
    parser.add_argument(
        "--monitor", action="store_true", help="Wrap env with gym.wrappers.Monitor."
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_rl_args()

    logging.basicConfig(level=args.log_level)

    if args.env not in FETCH_ENVS:
        raise Exception(f"Invalid environemt, please select from {FETCH_ENVS}")

    # Set a random seed used in PFRL.
    utils.set_random_seed(args.seed)

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2 ** 32

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    print("Output files are saved in {}".format(args.outdir))

    def make_env(test):
        env = gym.make(args.env)
        # Unwrap TimeLimit wrapper
        # fetch env is unique - it's conditioned on a goal
        assert isinstance(env, gym.wrappers.TimeLimit)
        env = env.env
        # Use different random seeds for train and test envs
        env_seed = 2 ** 32 - 1 - args.seed if test else args.seed
        env.seed(env_seed)
        # Cast observations to float32 because our model uses float32
        if args.monitor:
            env = pfrl.wrappers.Monitor(env, args.outdir)
        if args.render and not test:
            env = pfrl.wrappers.Render(env)
        return env

    def make_batch_env(test):
        return pfrl.envs.MultiprocessVectorEnv(
            [functools.partial(make_env, idx, test) for idx in range(args.num_envs)]
        )

    env = make_env(test=False)
    timestep_limit = env.spec.max_episode_steps
    obs_space_dict = env.observation_space
    action_space = env.action_space
    print("Observation space dictionary:", obs_space_dict)
    print("Action space:", action_space)

    # size of the subgoal is a hyperparameter
    env_subgoal_dim = 3
    # TODO - change the limits, they are completely wrong
    limits = np.array([0.2, 0.2, 0.2])
    subgoal_space = gym.spaces.Box(low=limits*-1, high=limits)

    env_state_dim = obs_space_dict.spaces['observation'].low.size
    env_goal_dim = obs_space_dict.spaces['desired_goal'].low.size
    env_action_dim = action_space.low.size

    scale_low = action_space.high * np.ones(env_action_dim)
    scale_high = subgoal_space.high * np.ones(env_subgoal_dim)

    def low_level_burnin_action_func():
        """Select random actions until model is updated one or more times."""
        return np.random.uniform(action_space.low, action_space.high).astype(np.float32)

    def high_level_burnin_action_func():
        """Select random actions until model is updated one or more times."""
        return np.random.uniform(subgoal_space.low, subgoal_space.high).astype(np.float32)

    gpu = 0 if torch.cuda.is_available() else None
    agent = HIROAgent(state_dim=env_state_dim,
                      action_dim=env_action_dim,
                      goal_dim=env_goal_dim,
                      subgoal_dim=env_subgoal_dim,
                      high_level_burnin_action_func=high_level_burnin_action_func,
                      low_level_burnin_action_func=low_level_burnin_action_func,
                      scale_low=scale_low,
                      scale_high=scale_high,
                      buffer_size=200000,
                      subgoal_freq=10,
                      train_freq=10,
                      reward_scaling=0.1,
                      goal_threshold=0.1,
                      gpu=gpu,
                      add_entropy=args.add_entropy)

    if args.load:
        # load weights from a file if arg supplied
        agent.load(args.load)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=env, agent=agent, n_steps=None, n_episodes=args.eval_n_runs
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
        # train the hierarchical agent

        experiments.train_hrl_agent_with_evaluation(
            agent=agent,
            env=make_env(test=False),
            steps=args.steps,
            outdir=args.outdir,
            eval_n_steps=None,
            eval_interval=5000,
            eval_n_episodes=10,
            use_tensorboard=True,
            train_max_episode_len=timestep_limit,
        )


if __name__ == "__main__":
    main()
