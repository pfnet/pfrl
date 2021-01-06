"""An example of training a Deep Recurrent Q-Network (DRQN).

DRQN is a DQN with a recurrent Q-network, described in
https://arxiv.org/abs/1507.06527.

To train DRQN for 50M timesteps on Breakout, run:
    python train_drqn_ale.py --recurrent

To train DQRN using a recurrent model on flickering 1-frame Breakout, run:
    python train_drqn_ale.py --recurrent --flicker --no-frame-stack
"""
import argparse

import gym
import gym.wrappers
import numpy as np
import torch
from torch import nn

import pfrl
from pfrl import experiments, explorers, replay_buffers, utils
from pfrl.q_functions import DiscreteActionValueHead
from pfrl.wrappers import atari_wrappers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default="BreakoutNoFrameskip-v4",
        help="OpenAI Atari domain to perform algorithm on.",
    )
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
    parser.add_argument("--demo", action="store_true", default=False)
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument(
        "--final-exploration-frames",
        type=int,
        default=10 ** 6,
        help="Timesteps after which we stop " + "annealing exploration rate",
    )
    parser.add_argument(
        "--final-epsilon",
        type=float,
        default=0.01,
        help="Final value of epsilon during training.",
    )
    parser.add_argument(
        "--eval-epsilon",
        type=float,
        default=0.001,
        help="Exploration epsilon used during eval episodes.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5 * 10 ** 7,
        help="Total number of timesteps to train the agent.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=30 * 60 * 60,  # 30 minutes with 60 fps
        help="Maximum number of frames for each episode.",
    )
    parser.add_argument(
        "--replay-start-size",
        type=int,
        default=5 * 10 ** 4,
        help="Minimum replay buffer size before " + "performing gradient updates.",
    )
    parser.add_argument(
        "--target-update-interval",
        type=int,
        default=3 * 10 ** 4,
        help="Frequency (in timesteps) at which " + "the target network is updated.",
    )
    parser.add_argument("--demo-n-episodes", type=int, default=30)
    parser.add_argument("--eval-n-steps", type=int, default=125000)
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=250000,
        help="Frequency (in timesteps) of evaluation phase.",
    )
    parser.add_argument(
        "--update-interval",
        type=int,
        default=4,
        help="Frequency (in timesteps) of network updates.",
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
    parser.add_argument(
        "--monitor",
        action="store_true",
        default=False,
        help=(
            "Monitor env. Videos and additional information are saved as output files."
        ),
    )
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate.")
    parser.add_argument(
        "--recurrent",
        action="store_true",
        default=False,
        help="Use a recurrent model. See the code for the model definition.",
    )
    parser.add_argument(
        "--flicker",
        action="store_true",
        default=False,
        help=(
            "Use so-called flickering Atari, where each"
            " screen is blacked out with probability 0.5."
        ),
    )
    parser.add_argument(
        "--no-frame-stack",
        action="store_true",
        default=False,
        help=(
            "Disable frame stacking so that the agent can only see the current screen."
        ),
    )
    parser.add_argument(
        "--episodic-update-len",
        type=int,
        default=10,
        help="Maximum length of sequences for updating recurrent models",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help=(
            "Number of transitions (in a non-recurrent case)"
            " or sequences (in a recurrent case) used for an"
            " update."
        ),
    )
    args = parser.parse_args()

    import logging

    logging.basicConfig(level=args.log_level)

    # Set a random seed used in PFRL.
    utils.set_random_seed(args.seed)

    # Set different random seeds for train and test envs.
    train_seed = args.seed
    test_seed = 2 ** 31 - 1 - args.seed

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    print("Output files are saved in {}".format(args.outdir))

    def make_env(test):
        # Use different random seeds for train and test envs
        env_seed = test_seed if test else train_seed
        env = atari_wrappers.wrap_deepmind(
            atari_wrappers.make_atari(args.env, max_frames=args.max_frames),
            episode_life=not test,
            clip_rewards=not test,
            flicker=args.flicker,
            frame_stack=not args.no_frame_stack,
        )
        env.seed(int(env_seed))
        if test:
            # Randomize actions like epsilon-greedy in evaluation as well
            env = pfrl.wrappers.RandomizeAction(env, args.eval_epsilon)
        if args.monitor:
            env = gym.wrappers.Monitor(
                env, args.outdir, mode="evaluation" if test else "training"
            )
        if args.render:
            env = pfrl.wrappers.Render(env)
        return env

    env = make_env(test=False)
    eval_env = make_env(test=True)
    print("Observation space", env.observation_space)
    print("Action space", env.action_space)

    n_frames = env.observation_space.shape[0]
    n_actions = env.action_space.n
    if args.recurrent:
        # Q-network with LSTM
        q_func = pfrl.nn.RecurrentSequential(
            nn.Conv2d(n_frames, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.Flatten(),
            nn.ReLU(),
            nn.LSTM(input_size=3136, hidden_size=512),
            nn.Linear(512, n_actions),
            DiscreteActionValueHead(),
        )
        # Replay buffer that stores whole episodes
        rbuf = replay_buffers.EpisodicReplayBuffer(10 ** 6)
    else:
        # Q-network without LSTM
        q_func = nn.Sequential(
            nn.Conv2d(n_frames, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
            DiscreteActionValueHead(),
        )
        # Replay buffer that stores transitions separately
        rbuf = replay_buffers.ReplayBuffer(10 ** 6)

    explorer = explorers.LinearDecayEpsilonGreedy(
        1.0,
        args.final_epsilon,
        args.final_exploration_frames,
        lambda: np.random.randint(n_actions),
    )

    opt = torch.optim.Adam(q_func.parameters(), lr=1e-4, eps=1e-4)

    def phi(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32) / 255

    agent = pfrl.agents.DoubleDQN(
        q_func,
        opt,
        rbuf,
        gpu=args.gpu,
        gamma=0.99,
        explorer=explorer,
        replay_start_size=args.replay_start_size,
        target_update_interval=args.target_update_interval,
        update_interval=args.update_interval,
        batch_accumulator="mean",
        phi=phi,
        minibatch_size=args.batch_size,
        episodic_update_len=args.episodic_update_len,
        recurrent=args.recurrent,
    )

    if args.load:
        agent.load(args.load)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=eval_env,
            agent=agent,
            n_steps=None,
            n_episodes=args.demo_n_episodes,
        )
        print(
            "n_runs: {} mean: {} median: {} stdev {}".format(
                args.demo_n_episodes,
                eval_stats["mean"],
                eval_stats["median"],
                eval_stats["stdev"],
            )
        )
    else:
        experiments.train_agent_with_evaluation(
            agent=agent,
            env=env,
            steps=args.steps,
            eval_n_steps=args.eval_n_steps,
            eval_n_episodes=None,
            eval_interval=args.eval_interval,
            outdir=args.outdir,
            eval_env=eval_env,
        )


if __name__ == "__main__":
    main()
