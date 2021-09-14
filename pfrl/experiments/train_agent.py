import csv
import logging
import os
import shutil
import time

from pfrl.experiments.evaluator import Evaluator, save_agent
from pfrl.utils.ask_yes_no import ask_yes_no


def save_agent_replay_buffer(agent, t, outdir, suffix="", logger=None):
    logger = logger or logging.getLogger(__name__)
    filename = os.path.join(outdir, "{}{}.replay.pkl".format(t, suffix))
    agent.replay_buffer.save(filename)
    logger.info("Saved the current replay buffer to %s", filename)


def ask_and_save_agent_replay_buffer(agent, t, outdir, suffix=""):
    if hasattr(agent, "replay_buffer") and ask_yes_no(
        "Replay buffer has {} transitions. Do you save them to a file?".format(
            len(agent.replay_buffer)
        )
    ):  # NOQA
        save_agent_replay_buffer(agent, t, outdir, suffix=suffix)


def snapshot(
    agent,
    t,
    episode_idx,
    outdir,
    suffix="_snapshot",
    logger=None,
    delete_old=True,
):
    start_time = time.time()
    tmp_suffix = f"{suffix}_"
    tmp_dirname = os.path.join(outdir, f"{t}{tmp_suffix}")  # use until files are saved
    agent.save(tmp_dirname)
    if hasattr(agent, "replay_buffer"):
        agent.replay_buffer.save(os.path.join(tmp_dirname, "replay.pkl"))
    if os.path.exists(os.path.join(outdir, "scores.txt")):
        shutil.copyfile(
            os.path.join(outdir, "scores.txt"), os.path.join(tmp_dirname, "scores.txt")
        )

    history_path = os.path.join(outdir, "snapshot_history.txt")
    if not os.path.exists(history_path):  # write header
        with open(history_path, "a") as f:
            csv.writer(f, delimiter="\t").writerow(["step", "episode", "snapshot_time"])
    with open(history_path, "a") as f:
        csv.writer(f, delimiter="\t").writerow(
            [t, episode_idx, time.time() - start_time]
        )
    shutil.copyfile(history_path, os.path.join(tmp_dirname, "snapshot_history.txt"))

    real_dirname = os.path.join(outdir, f"{t}{suffix}")
    os.rename(tmp_dirname, real_dirname)
    if logger:
        logger.info(f"Saved the snapshot to {real_dirname}")
    if delete_old:
        for old_dir in filter(
            lambda s: s.endswith(suffix) or s.endswith(tmp_suffix), os.listdir(outdir)
        ):
            if old_dir != f"{t}{suffix}":
                shutil.rmtree(os.path.join(outdir, old_dir))


def load_snapshot(agent, dirname, logger=None):
    agent.load(dirname)
    if hasattr(agent, "replay_buffer"):
        agent.replay_buffer.load(os.path.join(dirname, "replay.pkl"))
    if logger:
        logger.info(f"Loaded the snapshot from {dirname}")
    with open(os.path.join(dirname, "snapshot_history.txt")) as f:
        step, episode = map(int, f.readlines()[-1].split()[:2])
    max_score = None
    if os.path.exists(os.path.join(dirname, "scores.txt")):
        with open(os.path.join(dirname, "scores.txt")) as f:
            lines = f.readlines()
        if len(lines) > 1:
            max_score = float(lines[-1].split()[3])  # mean
    shutil.copyfile(
        os.path.join(dirname, "snapshot_history.txt"),
        os.path.join(dirname, "..", "snapshot_history.txt"),
    )
    shutil.copyfile(
        os.path.join(dirname, "scores.txt"),
        os.path.join(dirname, "..", "scores.txt"),
    )
    return step, episode, max_score


def latest_snapshot_dir(search_dir, suffix="_snapshot"):
    """
    return None if no snapshot exists
    """
    candidates = list(filter(lambda s: s.endswith(suffix), os.listdir(search_dir)))
    if len(candidates) == 0:
        return None
    return os.path.join(
        search_dir, max(candidates, key=lambda name: int(name.split("_")[0]))
    )


def train_agent(
    agent,
    env,
    steps,
    outdir,
    checkpoint_freq=None,
    take_resumable_snapshot=False,
    max_episode_len=None,
    step_offset=0,
    episode_offset=0,
    max_score=None,
    evaluator=None,
    successful_score=None,
    step_hooks=(),
    eval_during_episode=False,
    logger=None,
):

    logger = logger or logging.getLogger(__name__)

    # restore max_score
    if evaluator and max_score:
        evaluator.max_score = max_score

    episode_r = 0
    episode_idx = episode_offset

    # o_0, r_0
    obs = env.reset()

    t = step_offset
    if hasattr(agent, "t"):
        agent.t = step_offset

    eval_stats_history = []  # List of evaluation episode stats dict
    episode_len = 0
    try:
        while t < steps:

            # a_t
            action = agent.act(obs)
            # o_{t+1}, r_{t+1}
            obs, r, done, info = env.step(action)
            t += 1
            episode_r += r
            episode_len += 1
            reset = episode_len == max_episode_len or info.get("needs_reset", False)
            agent.observe(obs, r, done, reset)

            for hook in step_hooks:
                hook(env, agent, t)

            episode_end = done or reset or t == steps

            if episode_end:
                logger.info(
                    "outdir:%s step:%s episode:%s R:%s",
                    outdir,
                    t,
                    episode_idx,
                    episode_r,
                )
                stats = agent.get_statistics()
                logger.info("statistics:%s", stats)
                episode_idx += 1

            if evaluator is not None and (episode_end or eval_during_episode):
                eval_score = evaluator.evaluate_if_necessary(t=t, episodes=episode_idx)
                if eval_score is not None:
                    eval_stats = dict(agent.get_statistics())
                    eval_stats["eval_score"] = eval_score
                    eval_stats_history.append(eval_stats)
                if (
                    successful_score is not None
                    and evaluator.max_score >= successful_score
                ):
                    break

            if episode_end:
                if t == steps:
                    break
                # Start a new episode
                episode_r = 0
                episode_len = 0
                obs = env.reset()
            if checkpoint_freq and t % checkpoint_freq == 0:
                if take_resumable_snapshot:
                    snapshot(agent, t, episode_idx, outdir, logger=logger)
                else:
                    save_agent(agent, t, outdir, logger, suffix="_checkpoint")

    except (Exception, KeyboardInterrupt):
        # Save the current model before being killed
        save_agent(agent, t, outdir, logger, suffix="_except")
        raise

    # Save the final model
    save_agent(agent, t, outdir, logger, suffix="_finish")

    return eval_stats_history


def train_agent_with_evaluation(
    agent,
    env,
    steps,
    eval_n_steps,
    eval_n_episodes,
    eval_interval,
    outdir,
    checkpoint_freq=None,
    take_resumable_snapshot=False,
    train_max_episode_len=None,
    step_offset=0,
    episode_offset=0,
    eval_max_episode_len=None,
    max_score=None,
    eval_env=None,
    successful_score=None,
    step_hooks=(),
    evaluation_hooks=(),
    save_best_so_far_agent=True,
    use_tensorboard=False,
    eval_during_episode=False,
    logger=None,
):
    """Train an agent while periodically evaluating it.

    Args:
        agent: A pfrl.agent.Agent
        env: Environment train the agent against.
        steps (int): Total number of timesteps for training.
        eval_n_steps (int): Number of timesteps at each evaluation phase.
        eval_n_episodes (int): Number of episodes at each evaluation phase.
        eval_interval (int): Interval of evaluation.
        outdir (str): Path to the directory to output data.
        checkpoint_freq (int): frequency in step at which agents are stored.
        take_resumable_snapshot (bool): If True, snapshot is saved in checkpoint.
            Note that currently, snapshot does not support agent analytics (e.g.,
            for DQN, average_q, average_loss, cumulative_steps, and n_updates) and
            those valued in "scores.txt" might be incorrect after resuming from
            snapshot.
        train_max_episode_len (int): Maximum episode length during training.
        step_offset (int): Time step from which training starts.
        episode_offset (int): Episode index from which training starts,
        eval_max_episode_len (int or None): Maximum episode length of
            evaluation runs. If None, train_max_episode_len is used instead.
        max_score (int): Current max socre.
        eval_env: Environment used for evaluation.
        successful_score (float): Finish training if the mean score is greater
            than or equal to this value if not None
        step_hooks (Sequence): Sequence of callable objects that accepts
            (env, agent, step) as arguments. They are called every step.
            See pfrl.experiments.hooks.
        evaluation_hooks (Sequence): Sequence of
            pfrl.experiments.evaluation_hooks.EvaluationHook objects. They are
            called after each evaluation.
        save_best_so_far_agent (bool): If set to True, after each evaluation
            phase, if the score (= mean return of evaluation episodes) exceeds
            the best-so-far score, the current agent is saved.
        use_tensorboard (bool): Additionally log eval stats to tensorboard
        eval_during_episode (bool): Allow running evaluation during training episodes.
            This should be enabled only when `env` and `eval_env` are independent.
        logger (logging.Logger): Logger used in this function.
    Returns:
        agent: Trained agent.
        eval_stats_history: List of evaluation episode stats dict.
    """

    logger = logger or logging.getLogger(__name__)

    for hook in evaluation_hooks:
        if not hook.support_train_agent:
            raise ValueError(
                "{} does not support train_agent_with_evaluation().".format(hook)
            )

    os.makedirs(outdir, exist_ok=True)

    if eval_env is None:
        assert not eval_during_episode, (
            "To run evaluation during training episodes, you need to specify `eval_env`"
            " that is independent from `env`."
        )
        eval_env = env

    if eval_max_episode_len is None:
        eval_max_episode_len = train_max_episode_len

    evaluator = Evaluator(
        agent=agent,
        n_steps=eval_n_steps,
        n_episodes=eval_n_episodes,
        eval_interval=eval_interval,
        outdir=outdir,
        max_episode_len=eval_max_episode_len,
        env=eval_env,
        step_offset=step_offset,
        evaluation_hooks=evaluation_hooks,
        save_best_so_far_agent=save_best_so_far_agent,
        use_tensorboard=use_tensorboard,
        logger=logger,
    )

    eval_stats_history = train_agent(
        agent,
        env,
        steps,
        outdir,
        checkpoint_freq=checkpoint_freq,
        take_resumable_snapshot=take_resumable_snapshot,
        max_episode_len=train_max_episode_len,
        step_offset=step_offset,
        episode_offset=episode_offset,
        max_score=max_score,
        evaluator=evaluator,
        successful_score=successful_score,
        step_hooks=step_hooks,
        eval_during_episode=eval_during_episode,
        logger=logger,
    )

    return agent, eval_stats_history
