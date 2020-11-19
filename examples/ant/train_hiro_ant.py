"""
Training:
    python examples/ant/train_hiro_ant.py
Render Trained:
    python examples/ant/train_hiro_ant.py --render --demo --load <dir>
Example:
    python examples/ant/train_hiro_ant.py --render --demo --load results/6900d36edd696e65e1d2ae72dd58796a2d7c19ef-34c626fd-4418a6b0/best
"""
import argparse
import functools
import logging

import numpy as np
import torch

from hiro_robot_envs.envs import create_maze_env, AntEnvWithGoal

import pfrl
from pfrl import utils
from pfrl import experiments
from pfrl.agents.hrl.hiro_agent import HIROAgent


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
    parser.add_argument(
        "--record",
        action="store_true",
        default=False,
        help="Record videos of evaluation envs. --render should also be specified.",
    )
    parser.add_argument(
        "--env",
        default="AntMaze",
        help="Type of Ant Env to use. Options are AntMaze, AntFall, and AntPush.",
        type=str)
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Render env states in a GUI window.",
    )
    parser.add_argument("--num-envs", type=int, default=1, help="Number of envs run in parallel.")
    args = parser.parse_args()
    return args


def main():
    args = parse_rl_args()

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

    def make_ant_env(idx, test):

        # use different seeds for train vs test envs
        process_seed = int(process_seeds[idx])
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        # env_seed = np.random.randint(0, 2**32 - 1) if not test else process_seed
        utils.set_random_seed(env_seed)
        # create the anv environment with goal
        env = AntEnvWithGoal(create_maze_env(args.env), args.env, env_subgoal_dim=15)
        env.seed(int(env_seed))

        if args.render:
            env = pfrl.wrappers.GymLikeEnvRender(env)

        return env

    def make_batch_ant__env(test):
        return pfrl.envs.MultiprocessVectorEnv(
            [functools.partial(make_ant_env, idx, test) for idx in range(args.num_envs)]
        )

    eval_env = make_ant_env(0, test=True)

    env_state_dim = eval_env.state_dim
    env_action_dim = eval_env.action_dim

    env_subgoal_dim = eval_env.subgoal_dim

    # determined from the ant env
    if args.env == 'AntMaze' or args.env == 'AntPush':
        env_goal_dim = 2
    else:
        env_goal_dim = 3

    action_space = eval_env.action_space
    subgoal_space = eval_env.subgoal_space
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
                      gpu=gpu,
                      add_entropy=args.add_entropy)

    if args.load:
        # load weights from a file if arg supplied
        agent.load(args.load)

    if args.record:
        from mujoco_py import GlfwContext
        GlfwContext(offscreen=True)

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
        # train the hierarchical agent

        experiments.train_hrl_agent_with_evaluation(
            agent=agent,
            env=make_ant_env(0, test=False),
            steps=args.steps,
            outdir=args.outdir,
            eval_n_steps=None,
            eval_interval=5000,
            eval_n_episodes=10,
            use_tensorboard=True,
            record=args.record
        )


if __name__ == "__main__":
    main()
