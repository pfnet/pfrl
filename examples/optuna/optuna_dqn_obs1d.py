"""An example of training DQN, with Optuna-powered hyperparameters tuning.

An example script of training a DQN agent with
[Optuna](https://optuna.org/)-powered hyperparameter tuning.

To keep this script simple, the target environment (``--env``) must have:
 - 1d continuous observation space
 - discrete action space
The default arguments are set to LunarLander-v2 environment.
"""

import argparse
import logging
import os
import random

import gym
import torch.optim as optim

try:
    import optuna
except ImportError:
    raise RuntimeError("This script requires optuna installed.")

import pfrl
from pfrl import experiments, explorers, q_functions, replay_buffers, utils
from pfrl.agents.dqn import DQN
from pfrl.experiments.evaluation_hooks import OptunaPrunerHook
from pfrl.nn.mlp import MLP


def _objective_core(
    # optuna parameters
    trial,
    # training parameters
    env_id,
    outdir,
    seed,
    monitor,
    gpu,
    steps,
    train_max_episode_len,
    eval_n_episodes,
    eval_interval,
    batch_size,
    # hyperparameters
    hyperparams,
):
    # Set a random seed used in PFRL
    utils.set_random_seed(seed)

    # Set different random seeds for train and test envs.
    train_seed = seed
    test_seed = 2 ** 31 - 1 - seed

    def make_env(test=False):
        env = gym.make(env_id)

        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError(
                "Supported only Box observation environments, but given: {}".format(
                    env.observation_space
                )
            )
        if len(env.observation_space.shape) != 1:
            raise ValueError(
                "Supported only observation spaces with ndim==1, but given: {}".format(
                    env.observation_space.shape
                )
            )
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError(
                "Supported only discrete action environments, but given: {}".format(
                    env.action_space
                )
            )

        env_seed = test_seed if test else train_seed
        env.seed(env_seed)
        # Cast observations to float32 because our model uses float32
        env = pfrl.wrappers.CastObservationToFloat32(env)
        if monitor:
            env = pfrl.wrappers.Monitor(env, outdir)
        if not test:
            # Scale rewards (and thus returns) to a reasonable range so that
            # training is easier
            env = pfrl.wrappers.ScaleReward(env, hyperparams["reward_scale_factor"])
        return env

    env = make_env(test=False)
    obs_space = env.observation_space
    obs_size = obs_space.low.size
    action_space = env.action_space
    n_actions = action_space.n

    # create model & q_function
    model = MLP(
        in_size=obs_size, out_size=n_actions, hidden_sizes=hyperparams["hidden_sizes"]
    )
    q_func = q_functions.SingleModelStateQFunctionWithDiscreteAction(model=model)

    # Use epsilon-greedy for exploration
    start_epsilon = 1
    explorer = explorers.LinearDecayEpsilonGreedy(
        start_epsilon=start_epsilon,
        end_epsilon=hyperparams["end_epsilon"],
        decay_steps=hyperparams["decay_steps"],
        random_action_func=action_space.sample,
    )

    opt = optim.Adam(
        q_func.parameters(), lr=hyperparams["lr"], eps=hyperparams["adam_eps"]
    )

    rbuf_capacity = steps
    rbuf = replay_buffers.ReplayBuffer(rbuf_capacity)

    agent = DQN(
        q_func,
        opt,
        rbuf,
        gpu=gpu,
        gamma=hyperparams["gamma"],
        explorer=explorer,
        replay_start_size=hyperparams["replay_start_size"],
        target_update_interval=hyperparams["target_update_interval"],
        update_interval=hyperparams["update_interval"],
        minibatch_size=batch_size,
    )

    eval_env = make_env(test=True)

    evaluation_hooks = [OptunaPrunerHook(trial=trial)]
    _, eval_stats_history = experiments.train_agent_with_evaluation(
        agent=agent,
        env=env,
        steps=steps,
        eval_n_steps=None,
        eval_n_episodes=eval_n_episodes,
        eval_interval=eval_interval,
        outdir=outdir,
        eval_env=eval_env,
        train_max_episode_len=train_max_episode_len,
        evaluation_hooks=evaluation_hooks,
    )

    score = _get_score_from_eval_stats_history(eval_stats_history)

    return score


def _get_score_from_eval_stats_history(
    eval_stats_history, agg="last", target="eval_score"
):
    """Get a scalar score from a list of evaluation statistics dicts."""
    final_score = None
    if agg == "last":
        for stats in reversed(eval_stats_history):
            if target in stats:
                final_score = stats[target]
                break
    elif agg == "mean":
        scores = []
        for stats in eval_stats_history:
            if target in stats:
                score = stats[target]
                if score is not None:
                    scores.append(score)
        final_score = sum(scores) / len(scores)
    elif agg == "best":
        scores = []
        for stats in eval_stats_history:
            if target in stats:
                score = stats[target]
                if score is not None:
                    scores.append(score)
        final_score = max(scores)  # Assuming larger is better
    else:
        raise ValueError("Unknown agg method: {}".format(agg))

    if final_score is None:
        final_score = float("NaN")
    return final_score


def suggest(trial, steps):
    hyperparams = {}

    hyperparams["reward_scale_factor"] = trial.suggest_float(
        "reward_scale_factor", 1e-5, 10, log=True
    )
    n_hidden_layers = trial.suggest_int("n_hidden_layers", 1, 3)  # hyper-hyper-param
    hyperparams["hidden_sizes"] = []
    for n_channel in range(n_hidden_layers):
        # If n_channels is a large value, the precise number doesn't matter.
        # In other words, we should search over the smaller values more precisely.
        c = trial.suggest_int(
            "n_hidden_layers_{}_n_channels_{}".format(n_hidden_layers, n_channel),
            10,
            200,
            log=True,
        )
        hyperparams["hidden_sizes"].append(c)
    hyperparams["end_epsilon"] = trial.suggest_float("end_epsilon", 0.0, 0.3)
    max_decay_steps = steps // 2
    min_decay_steps = min(1e3, max_decay_steps)
    hyperparams["decay_steps"] = trial.suggest_int(
        "decay_steps", min_decay_steps, max_decay_steps
    )
    hyperparams["lr"] = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    # Adam's default eps==1e-8 but larger eps oftens helps.
    # (Rainbow: eps==1.5e-4, IQN: eps==1e-2/batch_size=3.125e-4)
    hyperparams["adam_eps"] = trial.suggest_float("adam_eps", 1e-8, 1e-3, log=True)
    inv_gamma = trial.suggest_float("inv_gamma", 1e-3, 1e-1, log=True)
    hyperparams["gamma"] = 1 - inv_gamma

    rbuf_capacity = steps
    min_replay_start_size = min(1e3, rbuf_capacity)
    # min: Replay start size cannot exceed replay buffer capacity.
    # max: decaying epsilon without training does not make much sense.
    max_replay_start_size = min(
        max(1e3, hyperparams["decay_steps"] // 2), rbuf_capacity
    )
    hyperparams["replay_start_size"] = trial.suggest_int(
        "replay_start_size",
        min_replay_start_size,
        max_replay_start_size,
    )
    # target_update_interval should be a multiple of update_interval
    hyperparams["update_interval"] = trial.suggest_int("update_interval", 1, 8)
    target_update_interval_coef = trial.suggest_int("target_update_interval_coef", 1, 4)
    hyperparams["target_update_interval"] = (
        hyperparams["update_interval"] * target_update_interval_coef
    )

    return hyperparams


def main():
    parser = argparse.ArgumentParser()

    # training parameters
    parser.add_argument(
        "--env",
        type=str,
        default="LunarLander-v2",
        help="OpenAI Gym Environment ID.",
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
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for randomizer.",
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
        default=10 ** 4,
        help="Frequency (in timesteps) of evaluation phase.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size.",
    )

    # Optuna related args
    parser.add_argument(
        "--optuna-study-name",
        type=str,
        default="optuna-pfrl-quickstart",
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
        "--optuna-training-steps-budget",
        type=int,
        default=4 * 10 ** 7,
        help=(
            "Total training steps thoughout the optimization. If the pruner works "
            "well, this limited training steps can be allocated to promissing trials "
            "efficiently, and thus the tuned hyperparameter should get better."
        ),
    )
    parser.add_argument(
        "--optuna-pruner",
        type=str,
        default="NopPruner",
        choices=["NopPruner", "ThresholdPruner", "PercentilePruner", "HyperbandPruner"],
        help=(
            "Optuna pruner. For more details see: "
            "https://optuna.readthedocs.io/en/stable/reference/pruners.html"
        ),
    )
    # add pruner specific arguments...
    _tmp_args, _unknown = parser.parse_known_args()
    n_warmup_steps_help_msg = (
        "Don't prune for first `n_warmup_steps` steps for each trial (pruning check "
        "will be invoked every `eval_interval` step). Note that `step` for the pruner "
        "is the training step, not the number of evaluations so far."
    )
    if _tmp_args.optuna_pruner == "NopPruner":
        pass
    elif _tmp_args.optuna_pruner == "ThresholdPruner":
        parser.add_argument(
            "--lower",
            type=float,
            required=True,
            help=(
                "Lower side threshold score for pruning trials. "
                "Please set the appropriate value for your specified env."
            ),
        )
        parser.add_argument(
            "--n-warmup-steps",
            type=int,
            default=5 * _tmp_args.eval_interval,
            help=n_warmup_steps_help_msg,
        )
    elif _tmp_args.optuna_pruner == "PercentilePruner":
        parser.add_argument(
            "--percentile",
            type=float,
            default=50.0,
            help="Setting percentile == 50.0 is equivalent to the MedianPruner.",
        )
        parser.add_argument(
            "--n-startup-trials",
            type=int,
            default=5,
        )
        parser.add_argument(
            "--n-warmup-steps",
            type=int,
            default=5 * _tmp_args.eval_interval,
            help=n_warmup_steps_help_msg,
        )
    elif _tmp_args.optuna_pruner == "HyperbandPruner":
        pass

    args = parser.parse_args()

    rootdir = experiments.prepare_output_dir(args=args, basedir=args.outdir)
    file_handler = logging.FileHandler(filename=os.path.join(rootdir, "console.log"))
    console_handler = logging.StreamHandler()
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])

    randomizer = random.Random(args.seed)

    def objective(trial):
        # suggest parameters from Optuna
        hyperparams = suggest(trial, args.steps)

        # seed is generated for each objective
        seed = randomizer.randint(0, 2 ** 31 - 1)
        additional_args = dict(seed=seed, **hyperparams)

        outdir = experiments.prepare_output_dir(args=additional_args, basedir=rootdir)
        print("Output files are saved in {}".format(outdir))

        return _objective_core(
            # optuna parameters
            trial=trial,
            # training parameters
            env_id=args.env,
            outdir=outdir,
            seed=seed,
            monitor=args.monitor,
            gpu=args.gpu,
            steps=args.steps,
            train_max_episode_len=args.train_max_episode_len,
            eval_n_episodes=args.eval_n_episodes,
            eval_interval=args.eval_interval,
            batch_size=args.batch_size,
            # hyperparameters
            hyperparams=hyperparams,
        )

    sampler = optuna.samplers.TPESampler(seed=args.seed)

    # pruner
    if args.optuna_pruner == "NopPruner":
        pruner = optuna.pruners.NopPruner()
    elif args.optuna_pruner == "ThresholdPruner":
        pruner = optuna.pruners.ThresholdPruner(
            lower=args.lower,
            n_warmup_steps=args.n_warmup_steps,
        )
    elif args.optuna_pruner == "PercentilePruner":
        pruner = optuna.pruners.PercentilePruner(
            percentile=args.percentile,
            n_startup_trials=args.n_startup_trials,
            n_warmup_steps=args.n_warmup_steps,
        )
    elif args.optuna_pruner == "HyperbandPruner":
        pruner = optuna.pruners.HyperbandPruner(min_resource=args.eval_interval)

    study = optuna.load_study(
        study_name=args.optuna_study_name,
        storage=args.optuna_storage,
        sampler=sampler,
        pruner=pruner,
    )

    class OptunaTrainingStepsBudgetCallback:
        def __init__(self, training_steps_budget, logger=None):
            self.training_steps_budget = training_steps_budget
            self.logger = logger or logging.getLogger(__name__)

        def __call__(self, study, trial):
            training_steps = sum(
                trial.last_step
                for trial in study.get_trials()
                if trial.last_step is not None
            )
            self.logger.info(
                "{} / {} (sum of training steps / budget)".format(
                    training_steps, self.training_steps_budget
                )
            )
            if training_steps >= self.training_steps_budget:
                study.stop()

    callbacks = [
        OptunaTrainingStepsBudgetCallback(
            training_steps_budget=args.optuna_training_steps_budget,
        ),
    ]
    study.optimize(objective, callbacks=callbacks)


if __name__ == "__main__":
    main()
