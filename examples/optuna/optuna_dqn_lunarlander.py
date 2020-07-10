"""An example of training DQN, with Optuna-powered hyper-parameters tuning.

An example script of training a DQN agent against the LunarLander-v2 environment,
with [Optuna](https://optuna.org/)-powered hyper-parameters tuning.
"""

import logging
import os
import argparse
import random

import torch.optim as optim
import gym
import optuna

import pfrl
from pfrl.agents.dqn import DQN
from pfrl import experiments
from pfrl import explorers
from pfrl import utils
from pfrl import q_functions
from pfrl import replay_buffers
from pfrl.nn.mlp import MLP


ENV_ID = "LunarLander-v2"


def _objective_core(
    # optuna parameters
    trial,
    # training parameters
    outdir,
    seed,
    monitor,
    gpu,
    steps,
    train_max_episode_len,
    eval_n_episodes,
    eval_interval,
    batch_size,
    # hyper parameters
    hyper_params,
):
    # Set a random seed used in PFRL
    utils.set_random_seed(seed)

    # Set different random seeds for train and test envs.
    train_seed = seed
    test_seed = 2 ** 31 - 1 - seed

    def make_env(test=False):
        env = gym.make(ENV_ID)
        env_seed = test_seed if test else train_seed
        env.seed(env_seed)
        # Cast observations to float32 because our model uses float32
        env = pfrl.wrappers.CastObservationToFloat32(env)
        if monitor:
            env = pfrl.wrappers.Monitor(env, outdir)
        if not test:
            # Scale rewards (and thus returns) to a reasonable range so that
            # training is easier
            env = pfrl.wrappers.ScaleReward(env, hyper_params["reward_scale_factor"])
        return env

    env = make_env(test=False)
    obs_space = env.observation_space
    obs_size = obs_space.low.size
    action_space = env.action_space
    n_actions = action_space.n

    # create model & q_function
    model = MLP(
        in_size=obs_size, out_size=n_actions, hidden_sizes=hyper_params["hidden_sizes"]
    )
    q_func = q_functions.SingleModelStateQFunctionWithDiscreteAction(model=model)

    # Use epsilon-greedy for exploration
    start_epsilon = 1
    explorer = explorers.LinearDecayEpsilonGreedy(
        start_epsilon=start_epsilon,
        end_epsilon=hyper_params["end_epsilon"],
        decay_steps=hyper_params["decay_steps"],
        random_action_func=action_space.sample,
    )

    opt = optim.Adam(
        q_func.parameters(), lr=hyper_params["lr"], eps=hyper_params["adam_eps"]
    )

    rbuf_capacity = steps
    rbuf = replay_buffers.ReplayBuffer(rbuf_capacity)

    agent = DQN(
        q_func,
        opt,
        rbuf,
        gpu=gpu,
        gamma=hyper_params["gamma"],
        explorer=explorer,
        replay_start_size=hyper_params["replay_start_size"],
        target_update_interval=hyper_params["target_update_interval"],
        update_interval=hyper_params["update_interval"],
        minibatch_size=batch_size,
    )

    eval_env = make_env(test=True)

    _, statistics = experiments.train_agent_with_evaluation(
        agent=agent,
        env=env,
        steps=steps,
        eval_n_steps=None,
        eval_n_episodes=eval_n_episodes,
        eval_interval=eval_interval,
        outdir=outdir,
        eval_env=eval_env,
        train_max_episode_len=train_max_episode_len,
    )

    score = _get_score_from_statistics(statistics)

    return score


def _get_score_from_statistics(statistics, agg="last", target="eval_score"):
    final_score = None
    if agg == "last":
        for stats in reversed(statistics):
            if target in stats:
                final_score = stats[target]
                break
    elif agg == "mean":
        scores = []
        for stats in statistics:
            if target in stats:
                score = stats[target]
                if score is not None:
                    scores.append(score)
        final_score = sum(scores) / len(scores)
    elif agg == "best":
        scores = []
        for stats in statistics:
            if target in stats:
                score = stats[target]
                if score is not None:
                    scores.append(score)
        final_score = max(scores)
    else:
        raise ValueError("Unknown agg method: {}".format(agg))

    if final_score is None:
        final_score = float("NaN")
    return final_score


def suggest(trial, steps):
    hyper_params = {}

    hyper_params["reward_scale_factor"] = trial.suggest_loguniform(
        "reward_scale_factor", 1e-5, 10
    )
    n_hidden_layers = trial.suggest_int("n_hidden_layers", 1, 3)  # hyper-hyper-param
    hyper_params["hidden_sizes"] = []
    for l in range(n_hidden_layers):
        # If n_channels is a large value, the precise number doesn't matter.
        # In other words, we should search over the smaller values more precisely.
        c = trial.suggest_loguniform(
            "n_hidden_layers_{}_n_channels_{}".format(n_hidden_layers, l), 10, 200
        )
        # But n_channels must be an integer.
        c = round(c)
        hyper_params["hidden_sizes"].append(c)
    hyper_params["end_epsilon"] = trial.suggest_uniform("end_epsilon", 0.0, 0.3)
    max_decay_steps = steps // 2
    hyper_params["decay_steps"] = trial.suggest_int("decay_steps", 1e3, max_decay_steps)
    hyper_params["lr"] = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    # Adam's default eps==1e-8 but larger eps oftens helps.
    # (Rainbow: eps==1.5e-4, IQN: eps==1e-2/batch_size=3.125e-4)
    hyper_params["adam_eps"] = trial.suggest_loguniform("adam_eps", 1e-8, 1e-3)
    inv_gamma = trial.suggest_loguniform("inv_gamma", 1e-3, 1e-1)
    hyper_params["gamma"] = 1 - inv_gamma
    # decaying epsilon without training does not make much sense.
    hyper_params["replay_start_size"] = trial.suggest_int(
        "replay_start_size", 1e3, max(1e3, hyper_params["decay_steps"] // 2),
    )
    # target_update_interval should be a multiple of update_interval
    hyper_params["update_interval"] = trial.suggest_int("update_interval", 1, 8)
    target_update_interval_coef = trial.suggest_int("target_update_interval_coef", 1, 4)
    hyper_params["target_update_interval"] = (
        hyper_params["update_interval"] * target_update_interval_coef
    )

    return hyper_params


def main():
    parser = argparse.ArgumentParser()

    # Optuna related args
    parser.add_argument(
        "--optuna-study-name",
        type=str,
        default="dqn_lunarlander",
        help="Name for Optuna Study.",
    )
    parser.add_argument(
        "--optuna-storage",
        type=str,
        default="sqlite:///example.db",
        help=(
            "DB URL for Optuna Study. Be sure to create one beforehand: "
            "optuna create-study --study-name <name> --storage <storage> --direction maximize"  # noqa
        ),
    )
    parser.add_argument(
        "--optuna-n-trials",
        type=int,
        default=100,
        help=(  # noqa
            "The number of trials for Optuna. See "
            "https://optuna.readthedocs.io/en/stable/reference/study.html#optuna.study.Study.optimize"  # noqa
        ),
    )

    # training parameters
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
        "--seed", type=int, default=0, help="Random seed for randomizer.",
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
        "--gpu", type=int, default=0, help="GPU to use, set to -1 if no GPU."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=4 * 10 ** 5,
        help="Total number of timesteps to train the agent for each trial",
    )
    parser.add_argument(
        "--train-max-episode-len",
        type=int,
        default=1000,
        help="Maximum episode length during training.",
    )
    parser.add_argument(
        "--eval-n-episodes",
        type=int,
        default=10,
        help="Number of episodes at each evaluation phase.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=4 * 10 ** 4,
        help="Frequency (in timesteps) of evaluation phase.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size.",
    )

    args = parser.parse_args()

    rootdir = experiments.prepare_output_dir(args=args, basedir=args.outdir)
    file_handler = logging.FileHandler(filename=os.path.join(rootdir, "console.log"))
    console_handler = logging.StreamHandler()
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])

    randomizer = random.Random(args.seed)

    def objective(trial):
        # suggest parameters from Optuna
        hyper_params = suggest(trial, args.steps)

        # seed is generated for each objective
        seed = randomizer.randint(0, 2 ** 31 - 1)
        additional_args = dict(seed=seed, **hyper_params)

        outdir = experiments.prepare_output_dir(args=additional_args, basedir=rootdir)
        print("Output files are saved in {}".format(outdir))

        return _objective_core(
            # optuna parameters
            trial=trial,
            # training parameters
            outdir=outdir,
            seed=seed,
            monitor=args.monitor,
            gpu=args.gpu,
            steps=args.steps,
            train_max_episode_len=args.train_max_episode_len,
            eval_n_episodes=args.eval_n_episodes,
            eval_interval=args.eval_interval,
            batch_size=args.batch_size,
            # hyper parameters
            hyper_params=hyper_params,
        )

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.load_study(
        study_name=args.optuna_study_name, storage=args.optuna_storage, sampler=sampler
    )
    study.optimize(objective, n_trials=args.optuna_n_trials)


if __name__ == "__main__":
    main()
