import argparse

import gym
import gym.spaces
import numpy as np
import torch
import torch.nn as nn

import pfrl
from pfrl import experiments, replay_buffers, utils
from pfrl.nn import BoundByTanh, ConcatObsAndAction
from pfrl.policies import DeterministicHead


class ComputeSuccessRate(gym.Wrapper):
    """Environment wrapper that computes success rate.

    Args:
        env: Env to wrap

    Attributes:
        success_record: list of successes
    """
    def __init__(self, env):
        super().__init__(env)
        self.success_record = []

    def reset(self):
        self.success_record.append(None)
        return self.env.reset()

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        assert "is_success" in info
        self.success_record[-1] = info["is_success"]
        return obs, r, done, info

    def get_statistics(self):
        # Ignore episodes with zero step
        valid_record = [x for x in self.success_record if x is not None]
        success_rate = (
            valid_record.count(True) / len(valid_record) if valid_record else np.nan
        )
        return [("success_rate", success_rate)]

    def clear_statistics(self):
        self.success_record = []


class ClipObservation(gym.ObservationWrapper):
    """Clip observations to a given range.

    Args:
        env: Env to wrap.
        low: Lower limit.
        high: Upper limit.

    Attributes:
        original_observation: Observation before casting.
    """

    def __init__(self, env, low, high):
        super().__init__(env)
        self.low = low
        self.high = high

    def observation(self, observation):
        self.original_observation = observation
        return np.clip(observation, self.low, self.high)


class EpsilonGreedyWithGaussianNoise(pfrl.explorer.Explorer):
    """Epsilon-Greedy with Gaussian noise.

    This type of explorer was used in 
    https://github.com/openai/baselines/tree/master/baselines/her
    """

    def __init__(self, epsilon, random_action_func, noise_scale, low=None, high=None):
        self.epsilon = epsilon
        self.random_action_func = random_action_func
        self.noise_scale = noise_scale
        self.low = low
        self.high = high

    def select_action(self, t, greedy_action_func, action_value=None):
        if np.random.rand() < self.epsilon:
            a = self.random_action_func()
        else:
            a = greedy_action_func()
            noise = np.random.normal(scale=self.noise_scale, size=a.shape).astype(
                np.float32
            )
            a = a + noise
        if self.low is not None or self.high is not None:
            return np.clip(a, self.low, self.high)
        else:
            return a

    def __repr__(self):
        return "EpsilonGreedyWithGaussianNoise(epsilon={}, noise_scale={}, low={}, high={})".format(
            self.epsilon, self.noise_scale, self.low, self.high
        )


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
    parser.add_argument(
        "--env",
        type=str,
        default="FetchReach-v1",
        help="OpenAI Gym MuJoCo env to perform algorithm on.",
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
        default=5 * 10 ** 3,
        help="Total number of timesteps to train the agent.",
    )
    parser.add_argument(
        "--replay-start-size",
        type=int,
        default=5 * 10 ** 2,
        help="Minimum replay buffer size before performing gradient updates.",
    )
    parser.add_argument("--replay-strategy",
                        default="future",
                        choices=["future", "final"],
                        help="The replay strategy to use",)
    parser.add_argument("--no-hindsight", action="store_true", default=False,
                        help="Do not use Hindsight Replay")
    parser.add_argument("--eval-n-episodes", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument(
        "--render", action="store_true", help="Render env states in a GUI window."
    )
    args = parser.parse_args()

    import logging

    logging.basicConfig(level=args.log_level)

    # Set a random seed used in PFRL.
    utils.set_random_seed(args.seed)

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    print("Output files are saved in {}".format(args.outdir))

    def make_env(test):
        env = gym.make(args.env)
        # Unwrap TimeLimit wrapper
        assert isinstance(env, gym.wrappers.TimeLimit)
        env = env.env
        # Use different random seeds for train and test envs
        env_seed = 2 ** 32 - 1 - args.seed if test else args.seed
        env.seed(env_seed)
        # Cast observations to float32 because our model uses float32
        if args.render and not test:
            env = pfrl.wrappers.Render(env)
        env = ComputeSuccessRate(env)
        return env

    env = make_env(test=False)
    timestep_limit = env.spec.max_episode_steps
    obs_space = env.observation_space
    action_space = env.action_space
    print("Observation space:", obs_space)
    print("Action space:", action_space)

    assert isinstance(obs_space, gym.spaces.Dict)
    obs_size = obs_space["observation"].low.size + obs_space["desired_goal"].low.size
    action_size = action_space.low.size

    def reward_fn(dg, ag):
        return env.compute_reward(ag, dg, None)

    q_func = nn.Sequential(
        ConcatObsAndAction(),
        nn.Linear(obs_size + action_size, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
    )
    policy = nn.Sequential(
        nn.Linear(obs_size, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, action_size),
        BoundByTanh(low=action_space.low, high=action_space.high),
        DeterministicHead(),
    )

    def init_xavier_uniform(layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    with torch.no_grad():
        q_func.apply(init_xavier_uniform)
        policy.apply(init_xavier_uniform)

    opt_a = torch.optim.Adam(policy.parameters())
    opt_c = torch.optim.Adam(q_func.parameters())

    if args.replay_strategy == "future":
        replay_strategy = replay_buffers.hindsight.ReplayFutureGoal()
    else:
        replay_strategy = replay_buffers.hindsight.ReplayFinalGoal()
    rbuf = replay_buffers.hindsight.HindsightReplayBuffer(
        reward_fn=reward_fn,
        replay_strategy=replay_strategy,
        capacity=10 ** 6,
    )

    explorer = EpsilonGreedyWithGaussianNoise(
        epsilon=0.3,
        random_action_func=lambda: env.action_space.sample(),
        noise_scale=0.2,
    )

    # Normalize observations based on their empirical mean and variance
    obs_normalizer = pfrl.nn.EmpiricalNormalization(obs_size, clip_threshold=5)

    def phi(observation):
        # Feature extractor
        obs = np.asarray(observation["observation"], dtype=np.float32)
        dg = np.asarray(observation["desired_goal"], dtype=np.float32)
        return np.concatenate((obs, dg)).clip(-200, 200)

    # 1 epoch = 10 episodes = 500 steps
    gamma = 1.0 - 1.0 / timestep_limit
    agent = pfrl.agents.DDPG(
        policy,
        q_func,
        opt_a,
        opt_c,
        rbuf,
        phi=phi,
        gamma=gamma,
        explorer=explorer,
        replay_start_size=256,
        target_update_method="soft",
        target_update_interval=50,
        update_interval=50,
        soft_update_tau=5e-2,
        n_times_update=40,
        gpu=args.gpu,
        minibatch_size=256,
        clip_return_range=(-1.0 / (1.0 - gamma), 0.0),
        action_l2_penalty_coef=1.0,
        obs_normalizer=obs_normalizer,
    )

    if args.load:
        agent.load(args.load)

    eval_env = make_env(test=True)
    if args.demo:
        eval_stats = experiments.eval_performance(
            env=eval_env,
            agent=agent,
            n_steps=args.eval_n_steps,
            n_episodes=None,
            max_episode_len=timestep_limit,
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
            train_max_episode_len=timestep_limit,
        )


if __name__ == "__main__":
    main()
