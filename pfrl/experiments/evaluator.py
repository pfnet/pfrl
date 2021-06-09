import logging
import multiprocessing as mp
import os
import statistics
import time

import numpy as np

import pfrl


def _run_episodes(
    env,
    agent,
    n_steps,
    n_episodes,
    max_episode_len=None,
    logger=None,
):
    """Run multiple episodes and return returns."""
    assert (n_steps is None) != (n_episodes is None)

    logger = logger or logging.getLogger(__name__)
    scores = []
    lengths = []
    terminate = False
    timestep = 0

    reset = True
    while not terminate:
        if reset:
            obs = env.reset()
            done = False
            test_r = 0
            episode_len = 0
            info = {}
        a = agent.act(obs)
        obs, r, done, info = env.step(a)
        test_r += r
        episode_len += 1
        timestep += 1
        reset = done or episode_len == max_episode_len or info.get("needs_reset", False)
        agent.observe(obs, r, done, reset)
        if reset:
            logger.info(
                "evaluation episode %s length:%s R:%s", len(scores), episode_len, test_r
            )
            # As mixing float and numpy float causes errors in statistics
            # functions, here every score is cast to float.
            scores.append(float(test_r))
            lengths.append(float(episode_len))
        if n_steps is None:
            terminate = len(scores) >= n_episodes
        else:
            terminate = timestep >= n_steps
    # If all steps were used for a single unfinished episode
    if len(scores) == 0:
        scores.append(float(test_r))
        lengths.append(float(episode_len))
        logger.info(
            "evaluation episode %s length:%s R:%s", len(scores), episode_len, test_r
        )
    return scores, lengths


def run_evaluation_episodes(
    env,
    agent,
    n_steps,
    n_episodes,
    max_episode_len=None,
    logger=None,
):
    """Run multiple evaluation episodes and return returns.

    Args:
        env (Environment): Environment used for evaluation
        agent (Agent): Agent to evaluate.
        n_steps (int): Number of timesteps to evaluate for.
        n_episodes (int): Number of evaluation runs.
        max_episode_len (int or None): If specified, episodes longer than this
            value will be truncated.
        logger (Logger or None): If specified, the given Logger object will be
            used for logging results. If not specified, the default logger of
            this module will be used.
    Returns:
        List of returns of evaluation runs.
    """
    with agent.eval_mode():
        return _run_episodes(
            env=env,
            agent=agent,
            n_steps=n_steps,
            n_episodes=n_episodes,
            max_episode_len=max_episode_len,
            logger=logger,
        )


def _batch_run_episodes(
    env,
    agent,
    n_steps,
    n_episodes,
    max_episode_len=None,
    logger=None,
):
    """Run multiple episodes and return returns in a batch manner."""
    assert (n_steps is None) != (n_episodes is None)

    logger = logger or logging.getLogger(__name__)
    num_envs = env.num_envs
    episode_returns = dict()
    episode_lengths = dict()
    episode_indices = np.zeros(num_envs, dtype="i")
    episode_idx = 0
    for i in range(num_envs):
        episode_indices[i] = episode_idx
        episode_idx += 1
    episode_r = np.zeros(num_envs, dtype=np.float64)
    episode_len = np.zeros(num_envs, dtype="i")

    obss = env.reset()
    rs = np.zeros(num_envs, dtype="f")

    termination_conditions = False
    timestep = 0
    while True:
        # a_t
        actions = agent.batch_act(obss)
        timestep += 1
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

        # Make mask. 0 if done/reset, 1 if pass
        end = np.logical_or(resets, dones)
        not_end = np.logical_not(end)

        for index in range(len(end)):
            if end[index]:
                episode_returns[episode_indices[index]] = episode_r[index]
                episode_lengths[episode_indices[index]] = episode_len[index]
                # Give the new episode an a new episode index
                episode_indices[index] = episode_idx
                episode_idx += 1

        episode_r[end] = 0
        episode_len[end] = 0

        # find first unfinished episode
        first_unfinished_episode = 0
        while first_unfinished_episode in episode_returns:
            first_unfinished_episode += 1

        # Check for termination conditions
        eval_episode_returns = []
        eval_episode_lens = []
        if n_steps is not None:
            total_time = 0
            for index in range(first_unfinished_episode):
                total_time += episode_lengths[index]
                # If you will run over allocated steps, quit
                if total_time > n_steps:
                    break
                else:
                    eval_episode_returns.append(episode_returns[index])
                    eval_episode_lens.append(episode_lengths[index])
            termination_conditions = total_time >= n_steps
            if not termination_conditions:
                unfinished_index = np.where(
                    episode_indices == first_unfinished_episode
                )[0]
                if total_time + episode_len[unfinished_index] >= n_steps:
                    termination_conditions = True
                    if first_unfinished_episode == 0:
                        eval_episode_returns.append(episode_r[unfinished_index])
                        eval_episode_lens.append(episode_len[unfinished_index])

        else:
            termination_conditions = first_unfinished_episode >= n_episodes
            if termination_conditions:
                # Get the first n completed episodes
                for index in range(n_episodes):
                    eval_episode_returns.append(episode_returns[index])
                    eval_episode_lens.append(episode_lengths[index])

        if termination_conditions:
            # If this is the last step, make sure the agent observes reset=True
            resets.fill(True)

        # Agent observes the consequences.
        agent.batch_observe(obss, rs, dones, resets)

        if termination_conditions:
            break
        else:
            obss = env.reset(not_end)

    for i, (epi_len, epi_ret) in enumerate(
        zip(eval_episode_lens, eval_episode_returns)
    ):
        logger.info("evaluation episode %s length: %s R: %s", i, epi_len, epi_ret)
    scores = [float(r) for r in eval_episode_returns]
    lengths = [float(ln) for ln in eval_episode_lens]
    return scores, lengths


def batch_run_evaluation_episodes(
    env,
    agent,
    n_steps,
    n_episodes,
    max_episode_len=None,
    logger=None,
):
    """Run multiple evaluation episodes and return returns in a batch manner.

    Args:
        env (VectorEnv): Environment used for evaluation.
        agent (Agent): Agent to evaluate.
        n_steps (int): Number of total timesteps to evaluate the agent.
        n_episodes (int): Number of evaluation runs.
        max_episode_len (int or None): If specified, episodes
            longer than this value will be truncated.
        logger (Logger or None): If specified, the given Logger
            object will be used for logging results. If not
            specified, the default logger of this module will
            be used.

    Returns:
        List of returns of evaluation runs.
    """
    with agent.eval_mode():
        return _batch_run_episodes(
            env=env,
            agent=agent,
            n_steps=n_steps,
            n_episodes=n_episodes,
            max_episode_len=max_episode_len,
            logger=logger,
        )


def eval_performance(
    env, agent, n_steps, n_episodes, max_episode_len=None, logger=None
):
    """Run multiple evaluation episodes and return statistics.

    Args:
        env (Environment): Environment used for evaluation
        agent (Agent): Agent to evaluate.
        n_steps (int): Number of timesteps to evaluate for.
        n_episodes (int): Number of evaluation episodes.
        max_episode_len (int or None): If specified, episodes longer than this
            value will be truncated.
        logger (Logger or None): If specified, the given Logger object will be
            used for logging results. If not specified, the default logger of
            this module will be used.
    Returns:
        Dict of statistics.
    """

    assert (n_steps is None) != (n_episodes is None)

    if isinstance(env, pfrl.env.VectorEnv):
        scores, lengths = batch_run_evaluation_episodes(
            env,
            agent,
            n_steps,
            n_episodes,
            max_episode_len=max_episode_len,
            logger=logger,
        )
    else:
        scores, lengths = run_evaluation_episodes(
            env,
            agent,
            n_steps,
            n_episodes,
            max_episode_len=max_episode_len,
            logger=logger,
        )
    stats = dict(
        episodes=len(scores),
        mean=statistics.mean(scores),
        median=statistics.median(scores),
        stdev=statistics.stdev(scores) if len(scores) >= 2 else 0.0,
        max=np.max(scores),
        min=np.min(scores),
        length_mean=statistics.mean(lengths),
        length_median=statistics.median(lengths),
        length_stdev=statistics.stdev(lengths) if len(lengths) >= 2 else 0,
        length_max=np.max(lengths),
        length_min=np.min(lengths),
    )
    return stats


def record_stats(outdir, values):
    with open(os.path.join(outdir, "scores.txt"), "a+") as f:
        print("\t".join(str(x) for x in values), file=f)


def create_tb_writer(outdir):
    """Return a tensorboard summarywriter with a custom scalar."""
    # This conditional import will raise an error if tensorboard<1.14
    from torch.utils.tensorboard import SummaryWriter

    tb_writer = SummaryWriter(log_dir=outdir)
    layout = {
        "Aggregate Charts": {
            "mean w/ min-max": [
                "Margin",
                ["eval/mean", "eval/min", "eval/max"],
            ],
            "mean +/- std": [
                "Margin",
                ["eval/mean", "extras/meanplusstdev", "extras/meanminusstdev"],
            ],
        }
    }
    tb_writer.add_custom_scalars(layout)
    return tb_writer


def record_tb_stats(summary_writer, agent_stats, eval_stats, env_stats, t):
    cur_time = time.time()

    for stat, value in agent_stats:
        summary_writer.add_scalar("agent/" + stat, value, t, cur_time)

    for stat, value in env_stats:
        summary_writer.add_scalar("env/" + stat, value, t, cur_time)

    for stat in ("mean", "median", "max", "min", "stdev"):
        value = eval_stats[stat]
        summary_writer.add_scalar("eval/" + stat, value, t, cur_time)

    summary_writer.add_scalar(
        "extras/meanplusstdev", eval_stats["mean"] + eval_stats["stdev"], t, cur_time
    )
    summary_writer.add_scalar(
        "extras/meanminusstdev", eval_stats["mean"] - eval_stats["stdev"], t, cur_time
    )

    # manually flush to avoid loosing events on termination
    summary_writer.flush()


def record_tb_stats_loop(outdir, queue, stop_event):
    tb_writer = create_tb_writer(outdir)

    while not (stop_event.wait(1e-6) and queue.empty()):
        if not queue.empty():
            agent_stats, eval_stats, env_stats, t = queue.get()
            record_tb_stats(tb_writer, agent_stats, eval_stats, env_stats, t)


def save_agent(agent, t, outdir, logger, suffix=""):
    dirname = os.path.join(outdir, "{}{}".format(t, suffix))
    agent.save(dirname)
    logger.info("Saved the agent to %s", dirname)


def write_header(outdir, agent, env):
    # Columns that describe information about an experiment.
    basic_columns = (
        "steps",  # number of time steps taken (= number of actions taken)
        "episodes",  # number of episodes finished
        "elapsed",  # time elapsed so far (seconds)
        "mean",  # mean of returns of evaluation runs
        "median",  # median of returns of evaluation runs
        "stdev",  # stdev of returns of evaluation runs
        "max",  # maximum value of returns of evaluation runs
        "min",  # minimum value of returns of evaluation runs
    )
    with open(os.path.join(outdir, "scores.txt"), "w") as f:
        custom_columns = tuple(t[0] for t in agent.get_statistics())
        env_get_stats = getattr(env, "get_statistics", lambda: [])
        assert callable(env_get_stats)
        custom_env_columns = tuple(t[0] for t in env_get_stats())
        column_names = basic_columns + custom_columns + custom_env_columns
        print("\t".join(column_names), file=f)


class Evaluator(object):
    """Object that is responsible for evaluating a given agent.

    Args:
        agent (Agent): Agent to evaluate.
        env (Env): Env to evaluate the agent on.
        n_steps (int): Number of timesteps used in each evaluation.
        n_episodes (int): Number of episodes used in each evaluation.
        eval_interval (int): Interval of evaluations in steps.
        outdir (str): Path to a directory to save things.
        max_episode_len (int): Maximum length of episodes used in evaluations.
        step_offset (int): Offset of steps used to schedule evaluations.
        evaluation_hooks (Sequence): Sequence of
            pfrl.experiments.evaluation_hooks.EvaluationHook objects. They are
            called after each evaluation.
        save_best_so_far_agent (bool): If set to True, after each evaluation,
            if the score (= mean of returns in evaluation episodes) exceeds
            the best-so-far score, the current agent is saved.
        use_tensorboard (bool): Additionally log eval stats to tensorboard
    """

    def __init__(
        self,
        agent,
        env,
        n_steps,
        n_episodes,
        eval_interval,
        outdir,
        max_episode_len=None,
        step_offset=0,
        evaluation_hooks=(),
        save_best_so_far_agent=True,
        logger=None,
        use_tensorboard=False,
    ):
        assert (n_steps is None) != (n_episodes is None), (
            "One of n_steps or n_episodes must be None. "
            + "Either we evaluate for a specified number "
            + "of episodes or for a specified number of timesteps."
        )
        self.agent = agent
        self.env = env
        self.max_score = np.finfo(np.float32).min
        self.start_time = time.time()
        self.n_steps = n_steps
        self.n_episodes = n_episodes
        self.eval_interval = eval_interval
        self.outdir = outdir
        self.use_tensorboard = use_tensorboard
        self.max_episode_len = max_episode_len
        self.step_offset = step_offset
        self.prev_eval_t = self.step_offset - self.step_offset % self.eval_interval
        self.evaluation_hooks = evaluation_hooks
        self.save_best_so_far_agent = save_best_so_far_agent
        self.logger = logger or logging.getLogger(__name__)
        self.env_get_stats = getattr(self.env, "get_statistics", lambda: [])
        self.env_clear_stats = getattr(self.env, "clear_statistics", lambda: None)
        assert callable(self.env_get_stats)
        assert callable(self.env_clear_stats)

        # Write a header line first
        write_header(self.outdir, self.agent, self.env)

        if use_tensorboard:
            self.tb_writer = create_tb_writer(outdir)

    def evaluate_and_update_max_score(self, t, episodes):
        self.env_clear_stats()
        eval_stats = eval_performance(
            self.env,
            self.agent,
            self.n_steps,
            self.n_episodes,
            max_episode_len=self.max_episode_len,
            logger=self.logger,
        )
        elapsed = time.time() - self.start_time
        agent_stats = self.agent.get_statistics()
        custom_values = tuple(tup[1] for tup in agent_stats)
        env_stats = self.env_get_stats()
        custom_env_values = tuple(tup[1] for tup in env_stats)
        mean = eval_stats["mean"]
        values = (
            (
                t,
                episodes,
                elapsed,
                mean,
                eval_stats["median"],
                eval_stats["stdev"],
                eval_stats["max"],
                eval_stats["min"],
            )
            + custom_values
            + custom_env_values
        )
        record_stats(self.outdir, values)

        if self.use_tensorboard:
            record_tb_stats(self.tb_writer, agent_stats, eval_stats, env_stats, t)

        for hook in self.evaluation_hooks:
            hook(
                env=self.env,
                agent=self.agent,
                evaluator=self,
                step=t,
                eval_stats=eval_stats,
                agent_stats=agent_stats,
                env_stats=env_stats,
            )

        if mean > self.max_score:
            self.logger.info("The best score is updated %s -> %s", self.max_score, mean)
            self.max_score = mean
            if self.save_best_so_far_agent:
                save_agent(self.agent, "best", self.outdir, self.logger)
        return mean

    def evaluate_if_necessary(self, t, episodes):
        if t >= self.prev_eval_t + self.eval_interval:
            score = self.evaluate_and_update_max_score(t, episodes)
            self.prev_eval_t = t - t % self.eval_interval
            return score
        return None


class AsyncEvaluator(object):
    """Object that is responsible for evaluating asynchronous multiple agents.

    Args:
        n_steps (int): Number of timesteps used in each evaluation.
        n_episodes (int): Number of episodes used in each evaluation.
        eval_interval (int): Interval of evaluations in steps.
        outdir (str): Path to a directory to save things.
        max_episode_len (int): Maximum length of episodes used in evaluations.
        step_offset (int): Offset of steps used to schedule evaluations.
        evaluation_hooks (Sequence): Sequence of
            pfrl.experiments.evaluation_hooks.EvaluationHook objects. They are
            called after each evaluation.
        save_best_so_far_agent (bool): If set to True, after each evaluation,
            if the score (= mean return of evaluation episodes) exceeds
            the best-so-far score, the current agent is saved.
    """

    def __init__(
        self,
        n_steps,
        n_episodes,
        eval_interval,
        outdir,
        max_episode_len=None,
        step_offset=0,
        evaluation_hooks=(),
        save_best_so_far_agent=True,
        logger=None,
    ):
        assert (n_steps is None) != (n_episodes is None), (
            "One of n_steps or n_episodes must be None. "
            + "Either we evaluate for a specified number "
            + "of episodes or for a specified number of timesteps."
        )
        self.start_time = time.time()
        self.n_steps = n_steps
        self.n_episodes = n_episodes
        self.eval_interval = eval_interval
        self.outdir = outdir
        self.max_episode_len = max_episode_len
        self.step_offset = step_offset
        self.evaluation_hooks = evaluation_hooks
        self.save_best_so_far_agent = save_best_so_far_agent
        self.logger = logger or logging.getLogger(__name__)

        # Values below are shared among processes
        self.prev_eval_t = mp.Value(
            "l", self.step_offset - self.step_offset % self.eval_interval
        )
        self._max_score = mp.Value("f", np.finfo(np.float32).min)
        self.wrote_header = mp.Value("b", False)

        # Create scores.txt
        with open(os.path.join(self.outdir, "scores.txt"), "a"):
            pass

        self.record_tb_stats_queue = None
        self.record_tb_stats_thread = None

    @property
    def max_score(self):
        with self._max_score.get_lock():
            v = self._max_score.value
        return v

    def evaluate_and_update_max_score(self, t, episodes, env, agent):
        env_get_stats = getattr(env, "get_statistics", lambda: [])
        env_clear_stats = getattr(env, "clear_statistics", lambda: None)
        assert callable(env_get_stats)
        assert callable(env_clear_stats)
        env_clear_stats()
        eval_stats = eval_performance(
            env,
            agent,
            self.n_steps,
            self.n_episodes,
            max_episode_len=self.max_episode_len,
            logger=self.logger,
        )
        elapsed = time.time() - self.start_time
        agent_stats = agent.get_statistics()
        custom_values = tuple(tup[1] for tup in agent_stats)
        env_stats = env_get_stats()
        custom_env_values = tuple(tup[1] for tup in env_stats)
        mean = eval_stats["mean"]
        values = (
            (
                t,
                episodes,
                elapsed,
                mean,
                eval_stats["median"],
                eval_stats["stdev"],
                eval_stats["max"],
                eval_stats["min"],
            )
            + custom_values
            + custom_env_values
        )
        record_stats(self.outdir, values)

        if self.record_tb_stats_queue is not None:
            self.record_tb_stats_queue.put([agent_stats, eval_stats, env_stats, t])

        for hook in self.evaluation_hooks:
            hook(
                env=env,
                agent=agent,
                evaluator=self,
                step=t,
                eval_stats=eval_stats,
                agent_stats=agent_stats,
                env_stats=env_stats,
            )

        with self._max_score.get_lock():
            if mean > self._max_score.value:
                self.logger.info(
                    "The best score is updated %s -> %s", self._max_score.value, mean
                )
                self._max_score.value = mean
                if self.save_best_so_far_agent:
                    save_agent(agent, "best", self.outdir, self.logger)
        return mean

    def evaluate_if_necessary(self, t, episodes, env, agent):
        necessary = False
        with self.prev_eval_t.get_lock():
            if t >= self.prev_eval_t.value + self.eval_interval:
                necessary = True
                self.prev_eval_t.value += self.eval_interval
        if necessary:
            with self.wrote_header.get_lock():
                if not self.wrote_header.value:
                    write_header(self.outdir, agent, env)
                    self.wrote_header.value = True
            return self.evaluate_and_update_max_score(t, episodes, env, agent)
        return None

    def start_tensorboard_writer(self, outdir, stop_event):
        self.record_tb_stats_queue = mp.Queue()
        self.record_tb_stats_thread = pfrl.utils.StoppableThread(
            target=record_tb_stats_loop,
            args=[outdir, self.record_tb_stats_queue, stop_event],
            stop_event=stop_event,
        )
        self.record_tb_stats_thread.start()

    def join_tensorboard_writer(self):
        self.record_tb_stats_thread.join()
