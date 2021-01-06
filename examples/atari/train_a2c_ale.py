import argparse
import functools
import logging

import numpy as np
from torch import nn

import pfrl
from pfrl import experiments, utils
from pfrl.agents import a2c
from pfrl.policies import SoftmaxCategoricalHead
from pfrl.wrappers import atari_wrappers


def phi(x):
    # Feature extractor
    return np.asarray(x, dtype=np.float32) / 255


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="BreakoutNoFrameskip-v4")
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 31)")
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=30 * 60 * 60,  # 30 minutes with 60 fps
        help="Maximum number of frames for each episode.",
    )
    parser.add_argument("--steps", type=int, default=8 * 10 ** 7)
    parser.add_argument("--update-steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--rmsprop-epsilon", type=float, default=1e-5)
    parser.add_argument(
        "--use-gae",
        action="store_true",
        default=False,
        help="use generalized advantage estimation",
    )
    parser.add_argument("--tau", type=float, default=0.95, help="gae parameter")
    parser.add_argument(
        "--alpha", type=float, default=0.99, help="RMSprop optimizer alpha"
    )
    parser.add_argument("--eval-interval", type=int, default=10 ** 6)
    parser.add_argument("--eval-n-runs", type=int, default=10)
    parser.add_argument("--demo", action="store_true", default=False)
    parser.add_argument("--load", type=str, default="")
    parser.add_argument(
        "--max-grad-norm", type=float, default=40, help="value loss coefficient"
    )
    parser.add_argument(
        "--gpu",
        "-g",
        type=int,
        default=-1,
        help="GPU ID (negative value indicates CPU)",
    )
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level. 10:DEBUG, 20:INFO etc.",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        default=False,
        help=(
            "Monitor env. Videos and additional information are saved as output files."
        ),
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Render env states in a GUI window.",
    )
    parser.set_defaults(use_lstm=False)
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    # Set a random seed used in PFRL.
    # If you use more than one processes, the results will be no longer
    # deterministic even with the same random seed.
    utils.set_random_seed(args.seed)

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2 ** 31

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    print("Output files are saved in {}".format(args.outdir))

    def make_env(process_idx, test):
        # Use different random seeds for train and test envs
        process_seed = process_seeds[process_idx]
        env_seed = 2 ** 31 - 1 - process_seed if test else process_seed
        env = atari_wrappers.wrap_deepmind(
            atari_wrappers.make_atari(args.env, max_frames=args.max_frames),
            episode_life=not test,
            clip_rewards=not test,
        )
        env.seed(int(env_seed))
        if args.monitor:
            env = pfrl.wrappers.Monitor(
                env, args.outdir, mode="evaluation" if test else "training"
            )
        if args.render:
            env = pfrl.wrappers.Render(env)
        return env

    def make_batch_env(test):
        return pfrl.envs.MultiprocessVectorEnv(
            [
                functools.partial(make_env, idx, test)
                for idx, env in enumerate(range(args.num_envs))
            ]
        )

    sample_env = make_env(0, test=False)
    obs_channel_size = sample_env.observation_space.low.shape[0]
    n_actions = sample_env.action_space.n

    model = nn.Sequential(
        nn.Conv2d(obs_channel_size, 16, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(2592, 256),
        nn.ReLU(),
        pfrl.nn.Branched(
            nn.Sequential(
                nn.Linear(256, n_actions),
                SoftmaxCategoricalHead(),
            ),
            nn.Linear(256, 1),
        ),
    )
    optimizer = pfrl.optimizers.RMSpropEpsInsideSqrt(
        model.parameters(),
        lr=args.lr,
        eps=args.rmsprop_epsilon,
        alpha=args.alpha,
    )

    agent = a2c.A2C(
        model,
        optimizer,
        gamma=args.gamma,
        gpu=args.gpu,
        num_processes=args.num_envs,
        update_steps=args.update_steps,
        phi=phi,
        use_gae=args.use_gae,
        tau=args.tau,
        max_grad_norm=args.max_grad_norm,
    )

    if args.load:
        agent.load(args.load)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=make_batch_env(test=True),
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs,
        )
        print(
            "n_runs: {} mean: {} median: {} stdev: {}".format(
                args.eval_n_runs,
                eval_stats["mean"],
                eval_stats["median"],
                eval_stats["stdev"],
            )
        )
    else:
        experiments.train_agent_batch_with_evaluation(
            agent=agent,
            env=make_batch_env(test=False),
            eval_env=make_batch_env(test=True),
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            outdir=args.outdir,
            save_best_so_far_agent=False,
            log_interval=1000,
        )


if __name__ == "__main__":
    main()
