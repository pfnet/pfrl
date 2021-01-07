import argparse
import os

# Prevent numpy from using multiple threads
os.environ["OMP_NUM_THREADS"] = "1"

import gym  # NOQA:E402
import gym.wrappers  # NOQA:E402
import numpy as np  # NOQA:E402
from torch import nn  # NOQA:E402

import pfrl  # NOQA:E402
from pfrl import experiments, utils  # NOQA:E402
from pfrl.agents import acer  # NOQA:E402
from pfrl.policies import SoftmaxCategoricalHead  # NOQA:E402
from pfrl.q_functions import DiscreteActionValueHead  # NOQA:E402
from pfrl.replay_buffers import EpisodicReplayBuffer  # NOQA:E402
from pfrl.wrappers import atari_wrappers  # NOQA:E402


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("processes", type=int)
    parser.add_argument("--env", type=str, default="BreakoutNoFrameskip-v4")
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 31)")
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help=(
            "Directory path to save output files."
            " If it does not exist, it will be created."
        ),
    )
    parser.add_argument("--t-max", type=int, default=5)
    parser.add_argument("--replay-start-size", type=int, default=10000)
    parser.add_argument("--n-times-replay", type=int, default=4)
    parser.add_argument("--beta", type=float, default=1e-2)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--steps", type=int, default=10 ** 7)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=30 * 60 * 60,  # 30 minutes with 60 fps
        help="Maximum number of frames for each episode.",
    )
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--eval-interval", type=int, default=10 ** 5)
    parser.add_argument("--eval-n-runs", type=int, default=10)
    parser.add_argument("--use-lstm", action="store_true")
    parser.add_argument("--demo", action="store_true", default=False)
    parser.add_argument("--load", type=str, default="")
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
    parser.add_argument(
        "--monitor",
        action="store_true",
        default=False,
        help=(
            "Monitor env. Videos and additional information are saved as output files."
        ),
    )
    parser.set_defaults(use_lstm=False)
    args = parser.parse_args()

    import logging

    logging.basicConfig(level=args.log_level)

    # Set a random seed used in PFRL.
    # If you use more than one processes, the results will be no longer
    # deterministic even with the same random seed.
    utils.set_random_seed(args.seed)

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.processes) + args.seed * args.processes
    assert process_seeds.max() < 2 ** 31

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    print("Output files are saved in {}".format(args.outdir))

    n_actions = gym.make(args.env).action_space.n

    input_to_hidden = nn.Sequential(
        nn.Conv2d(4, 16, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(2592, 256),
        nn.ReLU(),
    )

    head = acer.ACERDiscreteActionHead(
        pi=nn.Sequential(
            nn.Linear(256, n_actions),
            SoftmaxCategoricalHead(),
        ),
        q=nn.Sequential(
            nn.Linear(256, n_actions),
            DiscreteActionValueHead(),
        ),
    )

    if args.use_lstm:
        model = pfrl.nn.RecurrentSequential(
            input_to_hidden,
            nn.LSTM(num_layers=1, input_size=256, hidden_size=256),
            head,
        )
    else:
        model = nn.Sequential(input_to_hidden, head)

    model.apply(pfrl.initializers.init_chainer_default)

    opt = pfrl.optimizers.SharedRMSpropEpsInsideSqrt(
        model.parameters(), lr=args.lr, eps=4e-3, alpha=0.99
    )

    replay_buffer = EpisodicReplayBuffer(10 ** 6 // args.processes)

    def phi(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32) / 255

    agent = acer.ACER(
        model,
        opt,
        t_max=args.t_max,
        gamma=0.99,
        replay_buffer=replay_buffer,
        n_times_replay=args.n_times_replay,
        replay_start_size=args.replay_start_size,
        beta=args.beta,
        phi=phi,
        max_grad_norm=40,
        recurrent=args.use_lstm,
    )

    if args.load:
        agent.load(args.load)

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

    if args.demo:
        env = make_env(0, True)
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

        # Linearly decay the learning rate to zero
        def lr_setter(env, agent, value):
            for pg in agent.optimizer.param_groups:
                assert "lr" in pg
                pg["lr"] = value

        lr_decay_hook = experiments.LinearInterpolationHook(
            args.steps, args.lr, 0, lr_setter
        )

        experiments.train_agent_async(
            agent=agent,
            outdir=args.outdir,
            processes=args.processes,
            make_env=make_env,
            profile=args.profile,
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            global_step_hooks=[lr_decay_hook],
            save_best_so_far_agent=False,
        )


if __name__ == "__main__":
    main()
