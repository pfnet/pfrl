import logging
import os
import signal
import subprocess
import sys

import numpy as np
import torch
import torch.multiprocessing as mp
from torch import nn

from pfrl.experiments.evaluator import AsyncEvaluator
from pfrl.utils import async_, random_seed


def kill_all():
    if os.name == "nt":
        # windows
        # taskkill with /T kill all the subprocess
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(os.getpid())])
    else:
        pgid = os.getpgrp()
        os.killpg(pgid, signal.SIGTERM)
        sys.exit(1)


def train_loop(
    process_idx,
    env,
    agent,
    steps,
    outdir,
    counter,
    episodes_counter,
    stop_event,
    exception_event,
    max_episode_len=None,
    evaluator=None,
    eval_env=None,
    successful_score=None,
    logger=None,
    global_step_hooks=[],
):

    logger = logger or logging.getLogger(__name__)

    if eval_env is None:
        eval_env = env

    def save_model():
        if process_idx == 0:
            # Save the current model before being killed
            dirname = os.path.join(outdir, "{}_except".format(global_t))
            agent.save(dirname)
            logger.info("Saved the current model to %s", dirname)

    try:

        episode_r = 0
        global_t = 0
        local_t = 0
        global_episodes = 0
        obs = env.reset()
        episode_len = 0
        successful = False

        while True:

            # a_t
            a = agent.act(obs)
            # o_{t+1}, r_{t+1}
            obs, r, done, info = env.step(a)
            local_t += 1
            episode_r += r
            episode_len += 1
            reset = episode_len == max_episode_len or info.get("needs_reset", False)
            agent.observe(obs, r, done, reset)

            # Get and increment the global counter
            with counter.get_lock():
                counter.value += 1
                global_t = counter.value

            for hook in global_step_hooks:
                hook(env, agent, global_t)

            if done or reset or global_t >= steps or stop_event.is_set():
                if process_idx == 0:
                    logger.info(
                        "outdir:%s global_step:%s local_step:%s R:%s",
                        outdir,
                        global_t,
                        local_t,
                        episode_r,
                    )
                    logger.info("statistics:%s", agent.get_statistics())

                # Evaluate the current agent
                if evaluator is not None:
                    eval_score = evaluator.evaluate_if_necessary(
                        t=global_t, episodes=global_episodes, env=eval_env, agent=agent
                    )

                    if (
                        eval_score is not None
                        and successful_score is not None
                        and eval_score >= successful_score
                    ):
                        stop_event.set()
                        successful = True
                        # Break immediately in order to avoid an additional
                        # call of agent.act_and_train
                        break

                with episodes_counter.get_lock():
                    episodes_counter.value += 1
                    global_episodes = episodes_counter.value

                if global_t >= steps or stop_event.is_set():
                    break

                # Start a new episode
                episode_r = 0
                episode_len = 0
                obs = env.reset()

            if process_idx == 0 and exception_event.is_set():
                logger.exception("An exception detected, exiting")
                save_model()
                kill_all()

    except (Exception, KeyboardInterrupt):
        save_model()
        raise

    if global_t == steps:
        # Save the final model
        dirname = os.path.join(outdir, "{}_finish".format(steps))
        agent.save(dirname)
        logger.info("Saved the final agent to %s", dirname)

    if successful:
        # Save the successful model
        dirname = os.path.join(outdir, "successful")
        agent.save(dirname)
        logger.info("Saved the successful agent to %s", dirname)


def train_agent_async(
    outdir,
    processes,
    make_env,
    profile=False,
    steps=8 * 10 ** 7,
    eval_interval=10 ** 6,
    eval_n_steps=None,
    eval_n_episodes=10,
    eval_success_threshold=0.0,
    max_episode_len=None,
    step_offset=0,
    successful_score=None,
    agent=None,
    make_agent=None,
    global_step_hooks=[],
    evaluation_hooks=(),
    save_best_so_far_agent=True,
    use_tensorboard=False,
    logger=None,
    random_seeds=None,
    stop_event=None,
    exception_event=None,
):
    """Train agent asynchronously using multiprocessing.

    Either `agent` or `make_agent` must be specified.

    Args:
        outdir (str): Path to the directory to output things.
        processes (int): Number of processes.
        make_env (callable): (process_idx, test) -> Environment.
        profile (bool): Profile if set True.
        steps (int): Number of global time steps for training.
        eval_interval (int): Interval of evaluation. If set to None, the agent
            will not be evaluated at all.
        eval_n_steps (int): Number of eval timesteps at each eval phase
        eval_n_episodes (int): Number of eval episodes at each eval phase
        eval_success_threshold (float): r-threshold above which grasp succeeds
        max_episode_len (int): Maximum episode length.
        step_offset (int): Time step from which training starts.
        successful_score (float): Finish training if the mean score is greater
            or equal to this value if not None
        agent (Agent): Agent to train.
        make_agent (callable): (process_idx) -> Agent
        global_step_hooks (list): List of callable objects that accepts
            (env, agent, step) as arguments. They are called every global
            step. See pfrl.experiments.hooks.
        evaluation_hooks (Sequence): Sequence of
            pfrl.experiments.evaluation_hooks.EvaluationHook objects. They are
            called after each evaluation.
        save_best_so_far_agent (bool): If set to True, after each evaluation,
            if the score (= mean return of evaluation episodes) exceeds
            the best-so-far score, the current agent is saved.
        use_tensorboard (bool): Additionally log eval stats to tensorboard
        logger (logging.Logger): Logger used in this function.
        random_seeds (array-like of ints or None): Random seeds for processes.
            If set to None, [0, 1, ..., processes-1] are used.
        stop_event (multiprocessing.Event or None): Event to stop training.
            If set to None, a new Event object is created and used internally.
        exception_event (multiprocessing.Event or None): Event that indicates
            other thread raised an excpetion. The train will be terminated and
            the current agent will be saved.
            If set to None, a new Event object is created and used internally.

    Returns:
        Trained agent.
    """

    logger = logger or logging.getLogger(__name__)

    for hook in evaluation_hooks:
        if not hook.support_train_agent_async:
            raise ValueError("{} does not support train_agent_async().".format(hook))

    # Prevent numpy from using multiple threads
    os.environ["OMP_NUM_THREADS"] = "1"

    counter = mp.Value("l", 0)
    episodes_counter = mp.Value("l", 0)

    if stop_event is None:
        stop_event = mp.Event()

    if exception_event is None:
        exception_event = mp.Event()

    if agent is None:
        assert make_agent is not None
        agent = make_agent(0)

    # Move model and optimizer states in shared memory
    for attr in agent.shared_attributes:
        attr_value = getattr(agent, attr)
        if isinstance(attr_value, nn.Module):
            for k, v in attr_value.state_dict().items():
                v.share_memory_()
        elif isinstance(attr_value, torch.optim.Optimizer):
            for param, state in attr_value.state_dict()["state"].items():
                assert isinstance(state, dict)
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        v.share_memory_()

    if eval_interval is None:
        evaluator = None
    else:
        evaluator = AsyncEvaluator(
            n_steps=eval_n_steps,
            n_episodes=eval_n_episodes,
            eval_interval=eval_interval,
            outdir=outdir,
            max_episode_len=max_episode_len,
            step_offset=step_offset,
            evaluation_hooks=evaluation_hooks,
            save_best_so_far_agent=save_best_so_far_agent,
            logger=logger,
        )
        if use_tensorboard:
            evaluator.start_tensorboard_writer(outdir, stop_event)

    if random_seeds is None:
        random_seeds = np.arange(processes)

    def run_func(process_idx):
        random_seed.set_random_seed(random_seeds[process_idx])

        env = make_env(process_idx, test=False)
        if evaluator is None:
            eval_env = env
        else:
            eval_env = make_env(process_idx, test=True)
        if make_agent is not None:
            local_agent = make_agent(process_idx)
            for attr in agent.shared_attributes:
                setattr(local_agent, attr, getattr(agent, attr))
        else:
            local_agent = agent
        local_agent.process_idx = process_idx

        def f():
            train_loop(
                process_idx=process_idx,
                counter=counter,
                episodes_counter=episodes_counter,
                agent=local_agent,
                env=env,
                steps=steps,
                outdir=outdir,
                max_episode_len=max_episode_len,
                evaluator=evaluator,
                successful_score=successful_score,
                stop_event=stop_event,
                exception_event=exception_event,
                eval_env=eval_env,
                global_step_hooks=global_step_hooks,
                logger=logger,
            )

        if profile:
            import cProfile

            cProfile.runctx(
                "f()", globals(), locals(), "profile-{}.out".format(os.getpid())
            )
        else:
            f()

        env.close()
        if eval_env is not env:
            eval_env.close()

    async_.run_async(processes, run_func)

    stop_event.set()

    if evaluator is not None and use_tensorboard:
        evaluator.join_tensorboard_writer()

    return agent
