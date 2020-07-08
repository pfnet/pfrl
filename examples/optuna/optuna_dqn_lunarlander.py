"""An example of training DQN, with Optuna-powered hyper-parameters tuning.

An example script of training a DQN agent against the LunarLander-v2 environment,
with [Optuna](https://github.com/optuna/optuna)-powered hyper-parameters tuning.
"""

import logging
import os
import argparse
import random
import json

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

# meta parameters
ENV_ID = "LunarLander-v2"
BATCH_SIZE = 64  # should be tuned?
TRAIN_MAX_EPISODE_LEN = 1000
STEPS = 400 * TRAIN_MAX_EPISODE_LEN
EVAL_N_EPISODES = 3
EVAL_INTERVAL = STEPS // 10


def _objective_core(
    trial,
    # meta parameters
    outdir,
    seed,
    monitor,
    gpu,
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
    if hyper_params["explorer_args"]["explorer"] == "ExponentialDecayEpsilonGreedy":
        explorer = explorers.ExponentialDecayEpsilonGreedy(
            hyper_params["explorer_args"]["start_epsilon"],
            hyper_params["explorer_args"]["end_epsilon"],
            hyper_params["explorer_args"]["epsilon_decay"],
            action_space.sample,
        )
    elif hyper_params["explorer_args"]["explorer"] == "LinearDecayEpsilonGreedy":
        explorer = explorers.LinearDecayEpsilonGreedy(
            hyper_params["explorer_args"]["start_epsilon"],
            hyper_params["explorer_args"]["end_epsilon"],
            hyper_params["explorer_args"]["decay_steps"],
            action_space.sample,
        )
    else:
        raise ValueError(
            "Unknown explorer: {}".format(hyper_params["explorer_args"]["explorer"])
        )

    opt = optim.Adam(q_func.parameters(), lr=hyper_params["lr"])

    rbuf = replay_buffers.ReplayBuffer(hyper_params["rbuf_capacity"])

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
        minibatch_size=BATCH_SIZE,
    )

    eval_env = make_env(test=True)

    _, statistics = experiments.train_agent_with_evaluation(
        agent=agent,
        env=env,
        steps=STEPS,
        eval_n_steps=None,
        eval_n_episodes=EVAL_N_EPISODES,
        eval_interval=EVAL_INTERVAL,
        outdir=outdir,
        eval_env=eval_env,
        train_max_episode_len=TRAIN_MAX_EPISODE_LEN,
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


def _configure_logger(outdir):
    # We'd like to configure the logger in one process more than once...
    # Imitates `basicConfig(force=True)` behaviour introduced in Python3.8
    # See https://github.com/python/cpython/blob/3.8/Lib/logging/__init__.py#L1958-L1961
    root = logging.RootLogger(logging.WARNING)
    for h in root.handlers[:]:
        root.removeHandler(h)
        h.close()

    file_handler = logging.FileHandler(filename=os.path.join(outdir, "console.log"))
    console_handler = logging.StreamHandler()
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])


def suggest(trial):
    hyper_params = {}

    hyper_params["reward_scale_factor"] = trial.suggest_loguniform(
        "reward_scale_factor", 1e-5, 10
    )
    n_hidden_layers = trial.suggest_int("n_hidden_layers", 1, 3)  # hyper-hyper-param
    hyper_params["hidden_sizes"] = []
    for l in range(n_hidden_layers):
        c = trial.suggest_int(
            "n_hidden_layers_{}_n_channels_{}".format(n_hidden_layers, l), 10, 200
        )
        hyper_params["hidden_sizes"].append(c)
    hyper_params["explorer_args"] = {}
    hyper_params["explorer_args"]["explorer"] = trial.suggest_categorical(
        "explorer", ["ExponentialDecayEpsilonGreedy", "LinearDecayEpsilonGreedy"]
    )
    if hyper_params["explorer_args"]["explorer"] == "ExponentialDecayEpsilonGreedy":
        hyper_params["explorer_args"]["start_epsilon"] = trial.suggest_uniform(
            "ExponentialDecayEpsilonGreedy_start_epsilon", 0.5, 1.0
        )
        hyper_params["explorer_args"]["end_epsilon"] = trial.suggest_uniform(
            "ExponentialDecayEpsilonGreedy_end_epsilon", 0.0001, 0.3
        )
        hyper_params["explorer_args"]["epsilon_decay"] = trial.suggest_uniform(
            "ExponentialDecayEpsilonGreedy_epsilon_decay", 0.9, 0.999
        )
    elif hyper_params["explorer_args"]["explorer"] == "LinearDecayEpsilonGreedy":
        hyper_params["explorer_args"]["start_epsilon"] = trial.suggest_uniform(
            "LinearDecayEpsilonGreedy_start_epsilon", 0.5, 1.0
        )
        hyper_params["explorer_args"]["end_epsilon"] = trial.suggest_uniform(
            "LinearDecayEpsilonGreedy_end_epsilon", 0.0, 0.3
        )
        # low, high of this parameter is determined by
        # ExponentialDecayEpsilonGreedy's parameter
        # 0.5 * 0.9^low = 0.3
        # 1.0 * 0.999^high = 0.0001
        hyper_params["explorer_args"]["decay_steps"] = trial.suggest_int(
            "LinearDecayEpsilonGreedy_decay_steps", 5, 9206
        )
    hyper_params["lr"] = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    # note that the maximum training step size = 4e5
    hyper_params["rbuf_capacity"] = trial.suggest_int("rbuf_capacity", 1e4, 1e6)
    hyper_params["gamma"] = trial.suggest_uniform("gamma", 0.9, 1.0)
    hyper_params["replay_start_size"] = trial.suggest_int(
        "replay_start_size", BATCH_SIZE, 1e3
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
        help="The number of trials for Optuna. See https://optuna.readthedocs.io/en/stable/reference/study.html#optuna.study.Study.optimize",  # noqa
    )

    # PFRL related args
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
        "--gpu", type=int, default=0, help="GPU to use, set to -1 if no GPU."
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        default=False,
        help=(
            "Monitor env. Videos and additional information are saved as output files."
        ),
    )

    args = parser.parse_args()

    randomizer = random.Random(args.seed)

    def objective(trial):
        # setup meta paremeters...
        outdir = experiments.prepare_output_dir(args=args, basedir=args.outdir)
        print("Output files are saved in {}".format(outdir))
        _configure_logger(outdir)

        seed = randomizer.randint(0, 2 ** 31 - 1)

        # suggest parameters from Optuna
        hyper_params = suggest(trial)

        # seed is generated for each objective
        additional_args = dict(seed=seed, **hyper_params)
        with open(os.path.join(outdir, "additional_args.txt"), "w") as f:
            json.dump(additional_args, f)

        return _objective_core(
            trial,
            # meta parameters
            outdir=outdir,
            seed=seed,
            monitor=args.monitor,
            gpu=args.gpu,
            hyper_params=hyper_params,
        )

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.load_study(
        study_name=args.optuna_study_name, storage=args.optuna_storage, sampler=sampler
    )
    study.optimize(objective, n_trials=args.optuna_n_trials)


if __name__ == "__main__":
    main()
