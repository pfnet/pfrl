import logging
import os
from gym.spaces import space
from gym.spaces.space import Space
from pfrl.agents.hrl.hiro_agent import HIROAgent

from pfrl.experiments.evaluator import Evaluator
from pfrl.experiments.evaluator import save_agent


import os 
import argparse
import numpy as np
import datetime
import copy
from pfrl.utils import subgoal

from torch import sub
from gym import spaces


def train_hrl_agent(
    agent: HIROAgent,
    env,
    steps,
    outdir,
    checkpoint_freq=None,
    max_episode_len=None,
    step_offset=0,
    evaluator=None,
    successful_score=None,
    step_hooks=(),
    logger=None,
    explore=True,
    start_training_steps=100,
    subgoal=None
):

    logger = logger or logging.getLogger(__name__)

    episode_r = 0
    episode_idx = 0

    obs_dict = env.reset()
    subgoal = subgoal

    fg = obs_dict['desired_goal']
    obs = obs_dict['observation']

    # sample from subgoal
    sg = subgoal.sample()

    t = step_offset
    if hasattr(agent, "t"):
        agent.t = step_offset

    episode_len = 0
    try:
        while t < steps:
            # get action
            if explore:
                if t < start_training_steps:
                    action = env.action_space.sample()
                else:
                    action = agent.act_low_level(obs, sg)
            else:
                action = agent.act_low_level(obs, sg)

            # take a step in the environment
            obs_dict, r, done, info = env.step(action)
            obs = obs_dict['observation']

            if explore:
                if t < start_training_steps:
                    n_sg = subgoal.sample()
                else:
                    n_sg = agent.act_high_level(obs, fg, sg, t)
            else:
                n_sg = agent.act_high_level(obs, fg, sg, t)

            t += 1
            episode_r += r
            episode_len += 1

            reset = episode_len == max_episode_len or info.get("needs_reset", False)

            agent.observe(obs, fg, n_sg, r, done, reset, t, start_training_steps)
            sg = n_sg
            for hook in step_hooks:
                hook(env, agent, t)

            if done or reset or t == steps:
                logger.info(
                    "outdir:%s step:%s episode:%s R:%s",
                    outdir,
                    t,
                    episode_idx,
                    episode_r,
                )
                logger.info("statistics:%s", agent.get_statistics())
                if evaluator is not None:
                    evaluator.evaluate_if_necessary(t=t, episodes=episode_idx + 1)
                    if (
                        successful_score is not None
                        and evaluator.max_score >= successful_score
                    ):
                        break
                if t == steps:
                    break
                # Start a new episode
                episode_r = 0
                episode_idx += 1
                episode_len = 0
                obs_dict = env.reset()
                fg = obs_dict['desired_goal']
                obs = obs_dict['observation']
            if checkpoint_freq and t % checkpoint_freq == 0:
                save_agent(agent, t, outdir, logger, suffix="_checkpoint")

    except (Exception, KeyboardInterrupt):
        # Save the current model before being killed
        save_agent(agent, t, outdir, logger, suffix="_except")
        raise

    # Save the final model
    save_agent(agent, t, outdir, logger, suffix="_finish")


def train_hrl_agent_with_evaluation(
    agent,
    env,
    steps,
    eval_n_steps,
    eval_n_episodes,
    eval_interval,
    outdir,
    checkpoint_freq=None,
    train_max_episode_len=None,
    step_offset=0,
    eval_max_episode_len=None,
    eval_env=None,
    successful_score=None,
    step_hooks=(),
    save_best_so_far_agent=True,
    use_tensorboard=False,
    logger=None,
    subgoal=None
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
        checkpoint_freq (int): frequency at which agents are stored.
        train_max_episode_len (int): Maximum episode length during training.
        step_offset (int): Time step from which training starts.
        eval_max_episode_len (int or None): Maximum episode length of
            evaluation runs. If None, train_max_episode_len is used instead.
        eval_env: Environment used for evaluation.
        successful_score (float): Finish training if the mean score is greater
            than or equal to this value if not None
        step_hooks (Sequence): Sequence of callable objects that accepts
            (env, agent, step) as arguments. They are called every step.
            See pfrl.experiments.hooks.
        save_best_so_far_agent (bool): If set to True, after each evaluation
            phase, if the score (= mean return of evaluation episodes) exceeds
            the best-so-far score, the current agent is saved.
        use_tensorboard (bool): Additionally log eval stats to tensorboard
        logger (logging.Logger): Logger used in this function.
    """

    logger = logger or logging.getLogger(__name__)

    os.makedirs(outdir, exist_ok=True)

    if eval_env is None:
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
        save_best_so_far_agent=save_best_so_far_agent,
        use_tensorboard=use_tensorboard,
        logger=logger,
    )

    train_hrl_agent(
        agent,
        env,
        steps,
        outdir,
        checkpoint_freq=checkpoint_freq,
        max_episode_len=train_max_episode_len,
        step_offset=step_offset,
        evaluator=evaluator,
        successful_score=successful_score,
        step_hooks=step_hooks,
        logger=logger,
        subgoal=subgoal
    )


def run_evaluation(args, env, agent):
    agent.load(args.load_episode)

    rewards, success_rate = agent.evaluate_policy(env, args.eval_episodes, args.render, args.save_video, args.sleep)

    print('mean:{mean:.2f}, \
            std:{std:.2f}, \
            median:{median:.2f}, \
            success:{success:.2f}'.format(
                mean=np.mean(rewards), 
                std=np.std(rewards), 
                median=np.median(rewards), 
                success=success_rate))
