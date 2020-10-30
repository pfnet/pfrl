import argparse

import gym
import gym.spaces as spaces
import numpy as np
import torch
import torch.nn as nn

from pfrl.q_functions import DiscreteActionValueHead
from pfrl import agents
from pfrl import experiments
from pfrl import explorers
from pfrl import utils
from pfrl import replay_buffers

from pfrl.initializers import init_chainer_default


def reward_fn(dg, ag):
    return -1.0 if (ag != dg).any() else 0.0


class BitFlip(gym.GoalEnv):
    """BitFlip environment from https://arxiv.org/pdf/1707.01495.pdf

    Args:
        n: State space is {0,1}^n
    """

    def __init__(self, n):
        self.n = n
        self.action_space = spaces.Discrete(n)
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.MultiBinary(n),
                achieved_goal=spaces.MultiBinary(n),
                observation=spaces.MultiBinary(n),
            )
        )
        self.clear_statistics()

    def compute_reward(self, achieved_goal, desired_goal, info):
        return reward_fn(desired_goal, achieved_goal)

    def _check_done(self):
        success = (
            self.observation["desired_goal"] == self.observation["achieved_goal"]
        ).all()
        return (self.steps >= self.n) or success, success

    def step(self, action):
        # Compute action outcome
        bit_new = int(not self.observation["observation"][action])
        new_obs = self.observation["observation"].copy()
        new_obs[action] = bit_new
        # Set new observation
        dg = self.observation["desired_goal"]
        self.observation = {
            "desired_goal": dg.copy(),
            "achieved_goal": new_obs,
            "observation": new_obs,
        }

        reward = self.compute_reward(
            self.observation["achieved_goal"], self.observation["desired_goal"], {}
        )
        self.steps += 1
        done, success = self._check_done()
        assert success == (reward == 0)
        if done:
            result = 1 if success else 0
            self.results.append(result)
        return self.observation, reward, done, {}

    def reset(self):
        sample_obs = self.observation_space.sample()
        state, goal = sample_obs["observation"], sample_obs["desired_goal"]
        while (state == goal).all():
            sample_obs = self.observation_space.sample()
            state, goal = sample_obs["observation"], sample_obs["desired_goal"]
        self.observation = dict()
        self.observation["desired_goal"] = goal
        self.observation["achieved_goal"] = state
        self.observation["observation"] = state
        self.steps = 0
        return self.observation

    def get_statistics(self):
        failures = self.results.count(0)
        successes = self.results.count(1)
        assert len(self.results) == failures + successes
        if not self.results:
            return [("success_rate", None)]
        success_rate = successes / float(len(self.results))
        return [("success_rate", success_rate)]

    def clear_statistics(self):
        self.results = []


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
    parser.add_argument("--demo", action="store_true", default=False)
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level. 10:DEBUG, 20:INFO etc.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5 * 10 ** 6,
        help="Total number of timesteps to train the agent.",
    )
    parser.add_argument(
        "--replay-start-size",
        type=int,
        default=5 * 10 ** 2,
        help="Minimum replay buffer size before " + "performing gradient updates.",
    )
    parser.add_argument(
        "--num-bits",
        type=int,
        default=10,
        help="Number of bits for BitFlipping environment",
    )
    parser.add_argument("--no-hindsight", action="store_true", default=False)
    parser.add_argument("--eval-n-episodes", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=250000)
    parser.add_argument("--n-best-episodes", type=int, default=100)
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
        env = BitFlip(args.num_bits)
        env.seed(int(env_seed))
        return env

    env = make_env(test=False)
    eval_env = make_env(test=True)

    n_actions = env.action_space.n
    q_func = nn.Sequential(
        init_chainer_default(nn.Linear(args.num_bits * 2, 256)),
        nn.ReLU(),
        init_chainer_default(nn.Linear(256, n_actions)),
        DiscreteActionValueHead(),
    )

    opt = torch.optim.Adam(q_func.parameters(), eps=1e-4)

    if args.no_hindsight:
        rbuf = replay_buffers.ReplayBuffer(10 ** 6)
    else:
        rbuf = replay_buffers.hindsight.HindsightReplayBuffer(
            reward_fn=reward_fn,
            replay_strategy=replay_buffers.hindsight.ReplayFutureGoal(),
            capacity=10 ** 6,
        )

    decay_steps = (args.num_bits + 5) * 10 ** 3
    end_epsilon = min(0.1, 0.5 / args.num_bits)
    explorer = explorers.LinearDecayEpsilonGreedy(
        start_epsilon=0.5,
        end_epsilon=end_epsilon,
        decay_steps=decay_steps,
        random_action_func=lambda: np.random.randint(n_actions),
    )

    def phi(observation):
        # Feature extractor
        obs = np.asarray(observation["observation"], dtype=np.float32)
        dg = np.asarray(observation["desired_goal"], dtype=np.float32)
        return np.concatenate((obs, dg))

    Agent = agents.DoubleDQN
    agent = Agent(
        q_func,
        opt,
        rbuf,
        gpu=args.gpu,
        gamma=0.99,
        explorer=explorer,
        replay_start_size=args.replay_start_size,
        target_update_interval=10 ** 3,
        clip_delta=True,
        update_interval=4,
        batch_accumulator="sum",
        phi=phi,
    )

    if args.load:
        agent.load(args.load)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=eval_env, agent=agent, n_steps=args.eval_n_steps, n_episodes=None
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
