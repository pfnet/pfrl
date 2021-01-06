import argparse

import gym
import gym.spaces
import numpy as np
import torch
from torch import nn

import pfrl
from pfrl import agents, experiments, explorers
from pfrl import nn as pnn
from pfrl import replay_buffers, utils


class MultiBinaryAsDiscreteAction(gym.ActionWrapper):
    """Transforms MultiBinary action space to Discrete.

    If the action space of a given env is `gym.spaces.MultiBinary(n)`, then
    the action space of the wrapped env will be `gym.spaces.Discrete(2**n)`,
    which covers all the combinations of the original action space.

    Args:
        env (gym.Env): Gym env whose action space is `gym.spaces.MultiBinary`.
    """

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        self.orig_action_space = env.action_space
        self.action_space = gym.spaces.Discrete(2 ** env.action_space.n)

    def action(self, action):
        return [(action >> i) % 2 for i in range(self.orig_action_space.n)]


class DistributionalDuelingHead(nn.Module):
    """Head module for defining a distributional dueling network.

    This module expects a (batch_size, in_size)-shaped `torch.Tensor` as input
    and returns `pfrl.action_value.DistributionalDiscreteActionValue`.

    Args:
        in_size (int): Input size.
        n_actions (int): Number of actions.
        n_atoms (int): Number of atoms.
        v_min (float): Minimum value represented by atoms.
        v_max (float): Maximum value represented by atoms.
    """

    def __init__(self, in_size, n_actions, n_atoms, v_min, v_max):
        super().__init__()
        assert in_size % 2 == 0
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.register_buffer(
            "z_values", torch.linspace(v_min, v_max, n_atoms, dtype=torch.float)
        )
        self.a_stream = nn.Linear(in_size // 2, n_actions * n_atoms)
        self.v_stream = nn.Linear(in_size // 2, n_atoms)

    def forward(self, h):
        h_a, h_v = torch.chunk(h, 2, dim=1)
        a_logits = self.a_stream(h_a).reshape((-1, self.n_actions, self.n_atoms))
        a_logits = a_logits - a_logits.mean(dim=1, keepdim=True)
        v_logits = self.v_stream(h_v).reshape((-1, 1, self.n_atoms))
        probs = nn.functional.softmax(a_logits + v_logits, dim=2)
        return pfrl.action_value.DistributionalDiscreteActionValue(probs, self.z_values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="SlimeVolley-v0")
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help=(
            "Directory path to save output files."
            " If it does not exist, it will be created."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 32)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--demo", action="store_true", default=False)
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--noisy-net-sigma", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=2 * 10 ** 6)
    parser.add_argument("--replay-start-size", type=int, default=1600)
    parser.add_argument("--eval-n-episodes", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=250000)
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
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--v-max", type=float, default=1)
    parser.add_argument("--n-step-return", type=int, default=3)
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
        if "SlimeVolley" in args.env:
            # You need to install slimevolleygym
            import slimevolleygym  # NOQA

        env = gym.make(args.env)
        # Use different random seeds for train and test envs
        env_seed = test_seed if test else train_seed
        env.seed(int(env_seed))
        if args.monitor:
            env = pfrl.wrappers.Monitor(
                env, args.outdir, mode="evaluation" if test else "training"
            )
        if args.render:
            env = pfrl.wrappers.Render(env)
        if isinstance(env.action_space, gym.spaces.MultiBinary):
            env = MultiBinaryAsDiscreteAction(env)
        return env

    env = make_env(test=False)
    eval_env = make_env(test=True)

    obs_size = env.observation_space.low.size
    n_actions = env.action_space.n

    n_atoms = 51
    v_max = args.v_max
    v_min = -args.v_max
    hidden_size = 512
    q_func = nn.Sequential(
        nn.Linear(obs_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        DistributionalDuelingHead(hidden_size, n_actions, n_atoms, v_min, v_max),
    )

    def phi(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32)

    # Noisy nets
    pnn.to_factorized_noisy(q_func, sigma_scale=args.noisy_net_sigma)
    # Turn off explorer
    explorer = explorers.Greedy()

    # Use the same eps as https://arxiv.org/abs/1710.02298
    opt = torch.optim.Adam(q_func.parameters(), 1e-4, eps=1.5e-4)

    # Prioritized Replay
    # Anneal beta from beta0 to 1 throughout training
    update_interval = 1
    betasteps = args.steps / update_interval
    rbuf = replay_buffers.PrioritizedReplayBuffer(
        10 ** 6,
        alpha=0.5,
        beta0=0.4,
        betasteps=betasteps,
        num_steps=args.n_step_return,
        normalize_by_max="memory",
    )

    agent = agents.CategoricalDoubleDQN(
        q_func,
        opt,
        rbuf,
        gpu=args.gpu,
        gamma=args.gamma,
        explorer=explorer,
        minibatch_size=32,
        replay_start_size=args.replay_start_size,
        target_update_interval=2000,
        update_interval=update_interval,
        batch_accumulator="mean",
        phi=phi,
        max_grad_norm=10,
    )

    if args.load:
        agent.load(args.load)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=eval_env,
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_episodes,
        )
        print(
            "n_episodes: {} mean: {} median: {} stdev {}".format(
                eval_stats["episodes"],
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
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_episodes,
            eval_interval=args.eval_interval,
            outdir=args.outdir,
            save_best_so_far_agent=True,
            eval_env=eval_env,
        )


if __name__ == "__main__":
    main()
