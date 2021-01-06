import argparse
import functools
import os

import gym
import gym.spaces
import numpy as np
import torch
from torch import nn

import pfrl
from pfrl import experiments, explorers, replay_buffers, utils
from pfrl.q_functions import DiscreteActionValueHead


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
            (
                env.observation_space,
                gym.spaces.Discrete(self._max_steps + 1),
            )
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


class GraspingQFunction(nn.Module):
    """Q-function model for the grasping env.

    This model takes an 84x84 2D image and an integer that indicates the
    number of elapsed steps in an episode as input and outputs action values.
    """

    def __init__(self, n_actions, max_episode_steps):
        super().__init__()
        self.embed = nn.Embedding(max_episode_steps + 1, 3136)
        self.image2hidden = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.Flatten(),
        )
        self.hidden2out = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
            DiscreteActionValueHead(),
        )

    def forward(self, x):
        image, steps = x
        h = self.image2hidden(image) * torch.sigmoid(self.embed(steps))
        return self.hidden2out(h)


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

    def make_env(idx, test):
        from pybullet_envs.bullet.kuka_diverse_object_gym_env import (  # NOQA
            KukaDiverseObjectEnv,
        )

        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[idx])
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        # Set a random seed for this subprocess
        utils.set_random_seed(env_seed)
        env = KukaDiverseObjectEnv(
            isDiscrete=True,
            renders=args.render and (args.demo or not test),
            height=84,
            width=84,
            maxSteps=max_episode_steps,
            isTest=test,
        )
        # Disable file caching to keep memory usage small
        env._p.setPhysicsEngineParameter(enableFileCaching=False)
        assert env.observation_space is None
        env.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 3), dtype=np.uint8
        )
        # (84, 84, 3) -> (3, 84, 84)
        env = TransposeObservation(env, (2, 0, 1))
        env = ObserveElapsedSteps(env, max_episode_steps)
        # KukaDiverseObjectEnv internally asserts int actions
        env = CastAction(env, int)
        env.seed(int(env_seed))
        if test and args.record:
            assert args.render, "To use --record, --render needs be specified."
            video_dir = os.path.join(args.outdir, "video_{}".format(idx))
            os.mkdir(video_dir)
            env = RecordMovie(env, video_dir)
        return env

    def make_batch_env(test):
        return pfrl.envs.MultiprocessVectorEnv(
            [functools.partial(make_env, idx, test) for idx in range(args.num_envs)]
        )

    eval_env = make_batch_env(test=True)
    n_actions = eval_env.action_space.n

    q_func = GraspingQFunction(n_actions, max_episode_steps)

    # Use the hyper parameters of the Nature paper
    opt = pfrl.optimizers.RMSpropEpsInsideSqrt(
        q_func.parameters(),
        lr=args.lr,
        alpha=0.95,
        momentum=0.0,
        eps=1e-2,
        centered=True,
    )

    # Anneal beta from beta0 to 1 throughout training
    betasteps = args.steps / args.update_interval
    rbuf = replay_buffers.PrioritizedReplayBuffer(
        10 ** 6, alpha=0.6, beta0=0.4, betasteps=betasteps
    )

    explorer = explorers.LinearDecayEpsilonGreedy(
        1.0,
        args.final_epsilon,
        args.final_exploration_steps,
        lambda: np.random.randint(n_actions),
    )

    def phi(x):
        # Feature extractor
        image, elapsed_steps = x
        # Normalize RGB values: [0, 255] -> [0, 1]
        norm_image = np.asarray(image, dtype=np.float32) / 255
        return norm_image, elapsed_steps

    agent = pfrl.agents.DoubleDQN(
        q_func,
        opt,
        rbuf,
        gpu=args.gpu,
        gamma=args.gamma,
        explorer=explorer,
        minibatch_size=args.batch_size,
        replay_start_size=args.replay_start_size,
        target_update_interval=args.target_update_interval,
        update_interval=args.update_interval,
        batch_accumulator="sum",
        phi=phi,
    )

    if args.load:
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
        experiments.train_agent_batch_with_evaluation(
            agent=agent,
            env=make_batch_env(test=False),
            eval_env=eval_env,
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
