import logging
import os
from collections import deque

import numpy as np

from pfrl.experiments.evaluator import Evaluator, save_agent


def train_agent_batch(
    agent,
    env,
    steps,
    outdir,
    checkpoint_freq=None,
    log_interval=None,
    max_episode_len=None,
    step_offset=0,
    evaluator=None,
    successful_score=None,
    step_hooks=(),
    evaluation_hooks=(),
    return_window_size=100,
    logger=None,
):
    """Train an agent in a batch environment.

    Args:
        agent: Agent to train.
        env: Environment to train the agent against.
        steps (int): Number of total time steps for training.
        outdir (str): Path to the directory to output things.
        checkpoint_freq (int): frequency at which agents are stored.
        log_interval (int): Interval of logging.
        max_episode_len (int): Maximum episode length.
        step_offset (int): Time step from which training starts.
        return_window_size (int): Number of training episodes used to estimate
            the average returns of the current agent.
        successful_score (float): Finish training if the mean score is greater
            or equal to thisvalue if not None
        step_hooks (Sequence): Sequence of callable objects that accepts
            (env, agent, step) as arguments. They are called every step.
            See pfrl.experiments.hooks.
        evaluation_hooks (Sequence): Sequence of callable objects that accepts
            (env, agent, evaluator, step, eval_score) as arguments. They are
            called every evaluation. See pfrl.experiments.evaluation_hooks.
        logger (logging.Logger): Logger used in this function.
    Returns:
        List of evaluation episode stats dict.
    """

    logger = logger or logging.getLogger(__name__)
    recent_returns = deque(maxlen=return_window_size)

    num_envs = env.num_envs
    episode_r = np.zeros(num_envs, dtype=np.float64)
    episode_idx = np.zeros(num_envs, dtype="i")
    episode_len = np.zeros(num_envs, dtype="i")

    # o_0, r_0
    obss = env.reset()

    t = step_offset
    if hasattr(agent, "t"):
        agent.t = step_offset

    eval_stats_history = []  # List of evaluation episode stats dict
    try:
        while True:
            # a_t
            actions = agent.batch_act(obss)
            # o_{t+1}, r_{t+1}
            obss, rs, dones, infos = env.step(actions)
            episode_r += rs
            episode_len += 1

            # Compute mask for done and reset
            if max_episode_len is None:
                resets = np.zeros(num_envs, dtype=bool)
            else:
                resets = episode_len == max_episode_len
            resets = np.logical_or(
                resets, [info.get("needs_reset", False) for info in infos]
            )
            # Agent observes the consequences
            agent.batch_observe(obss, rs, dones, resets)

            # Make mask. 0 if done/reset, 1 if pass
            end = np.logical_or(resets, dones)
            not_end = np.logical_not(end)

            # For episodes that ends, do the following:
            #   1. increment the episode count
            #   2. record the return
            #   3. clear the record of rewards
            #   4. clear the record of the number of steps
            #   5. reset the env to start a new episode
            # 3-5 are skipped when training is already finished.
            episode_idx += end
            recent_returns.extend(episode_r[end])

            for _ in range(num_envs):
                t += 1
                if checkpoint_freq and t % checkpoint_freq == 0:
                    save_agent(agent, t, outdir, logger, suffix="_checkpoint")

                for hook in step_hooks:
                    hook(env, agent, t)

            if (
                log_interval is not None
                and t >= log_interval
                and t % log_interval < num_envs
            ):
                logger.info(
                    "outdir:{} step:{} episode:{} last_R: {} average_R:{}".format(  # NOQA
                        outdir,
                        t,
                        np.sum(episode_idx),
                        recent_returns[-1] if recent_returns else np.nan,
                        np.mean(recent_returns) if recent_returns else np.nan,
                    )
                )
                logger.info("statistics: {}".format(agent.get_statistics()))
            if evaluator:
                eval_score = evaluator.evaluate_if_necessary(
                    t=t, episodes=np.sum(episode_idx)
                )
                if eval_score is not None:
                    eval_stats = dict(agent.get_statistics())
                    eval_stats["eval_score"] = eval_score
                    eval_stats_history.append(eval_stats)
                    for hook in evaluation_hooks:
                        hook(env, agent, evaluator, t, eval_score)
                    if (
                        successful_score is not None
                        and evaluator.max_score >= successful_score
                    ):
                        break

            if t >= steps:
                break

            # Start new episodes if needed
            episode_r[end] = 0
            episode_len[end] = 0
            obss = env.reset(not_end)

    except (Exception, KeyboardInterrupt):
        # Save the current model before being killed
        save_agent(agent, t, outdir, logger, suffix="_except")
        env.close()
        if evaluator:
            evaluator.env.close()
        raise
    else:
        # Save the final model
        save_agent(agent, t, outdir, logger, suffix="_finish")

    return eval_stats_history


def train_agent_batch_with_evaluation(
    agent,
    env,
    steps,
    eval_n_steps,
    eval_n_episodes,
    eval_interval,
    outdir,
    checkpoint_freq=None,
    max_episode_len=None,
    step_offset=0,
    eval_max_episode_len=None,
    return_window_size=100,
    eval_env=None,
    log_interval=None,
    successful_score=None,
    step_hooks=(),
    evaluation_hooks=(),
    save_best_so_far_agent=True,
    logger=None,
):
    """Train an agent while regularly evaluating it.

    Args:
        agent: Agent to train.
        env: Environment train the againt against.
        steps (int): Number of total time steps for training.
        eval_n_steps (int): Number of timesteps at each evaluation phase.
        eval_n_runs (int): Number of runs for each time of evaluation.
        eval_interval (int): Interval of evaluation.
        outdir (str): Path to the directory to output things.
        log_interval (int): Interval of logging.
        checkpoint_freq (int): frequency with which to store networks
        max_episode_len (int): Maximum episode length.
        step_offset (int): Time step from which training starts.
        return_window_size (int): Number of training episodes used to estimate
            the average returns of the current agent.
        eval_max_episode_len (int or None): Maximum episode length of
            evaluation runs. If set to None, max_episode_len is used instead.
        eval_env: Environment used for evaluation.
        successful_score (float): Finish training if the mean score is greater
            or equal to thisvalue if not None
        step_hooks (Sequence): Sequence of callable objects that accepts
            (env, agent, step) as arguments. They are called every step.
            See pfrl.experiments.hooks.
        evaluation_hooks (Sequence): Sequence of callable objects that accepts
            (env, agent, evaluator, step, eval_score) as arguments. They are
            called every evaluation. See pfrl.experiments.evaluation_hooks.
        save_best_so_far_agent (bool): If set to True, after each evaluation,
            if the score (= mean return of evaluation episodes) exceeds
            the best-so-far score, the current agent is saved.
        logger (logging.Logger): Logger used in this function.
    Returns:
        agent: Trained agent.
        eval_stats_history: List of evaluation episode stats dict.
    """

    logger = logger or logging.getLogger(__name__)

    os.makedirs(outdir, exist_ok=True)

    if eval_env is None:
        eval_env = env

    if eval_max_episode_len is None:
        eval_max_episode_len = max_episode_len

    evaluator = Evaluator(
        agent=agent,
        n_steps=eval_n_steps,
        n_episodes=eval_n_episodes,
        eval_interval=eval_interval,
        outdir=outdir,
        max_episode_len=eval_max_episode_len,
        env=eval_env,
        step_offset=step_offset,
        save_best_so_far_agent=save_best_so_far_agent,
        logger=logger,
    )

    eval_stats_history = train_agent_batch(
        agent,
        env,
        steps,
        outdir,
        checkpoint_freq=checkpoint_freq,
        max_episode_len=max_episode_len,
        step_offset=step_offset,
        evaluator=evaluator,
        successful_score=successful_score,
        return_window_size=return_window_size,
        log_interval=log_interval,
        step_hooks=step_hooks,
        evaluation_hooks=evaluation_hooks,
        logger=logger,
    )

    return agent, eval_stats_history
