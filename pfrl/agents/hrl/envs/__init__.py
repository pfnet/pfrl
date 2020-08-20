"""Random policy on an environment."""

import numpy as np
import argparse

# import envs.create_maze_env
from . import create_maze_env


def get_goal_sample_fn(env_name, evaluate):
    if env_name == 'AntMaze':
        # NOTE: When evaluating (i.e. the metrics shown in the paper,
        # we use the commented out goal sampling function.    The uncommented
        # one is only used for training.
        if evaluate:
            return lambda: np.array([0., 16.])
        else:
            return lambda: np.random.uniform((-4, -4), (20, 20))
    elif env_name == 'AntPush':
        return lambda: np.array([0., 19.])
    elif env_name == 'AntFall':
        return lambda: np.array([0., 27., 4.5])
    else:
        assert False, 'Unknown env'


def get_reward_fn(env_name):
    if env_name == 'AntMaze':
        return lambda obs, goal: -np.sum(np.square(obs[:2] - goal)) ** 0.5
    elif env_name == 'AntPush':
        return lambda obs, goal: -np.sum(np.square(obs[:2] - goal)) ** 0.5
    elif env_name == 'AntFall':
        return lambda obs, goal: -np.sum(np.square(obs[:3] - goal)) ** 0.5
    else:
        assert False, 'Unknown env'


def success_fn(last_reward):
    return last_reward > -5.0


class EnvWithGoal(object):
    def __init__(self, base_env, env_name):
        self.base_env = base_env
        self.env_name = env_name
        self.evaluate = False
        self.reward_fn = get_reward_fn(env_name)
        self.goal = None
        self.distance_threshold = 5
        self.count = 0
        self.state_dim = self.base_env.observation_space.shape[0] + 1
        self.action_dim = self.base_env.action_space.shape[0]

    def seed(self, seed):
        self.base_env.seed(seed)

    def reset(self):
        # self.viewer_setup()
        self.goal_sample_fn = get_goal_sample_fn(self.env_name, self.evaluate)
        obs = self.base_env.reset()
        self.count = 0
        self.goal = self.goal_sample_fn()
        return {
            # add timestep
            'observation': np.r_[obs.copy(), self.count], 
            'achieved_goal': obs[:2],
            'desired_goal': self.goal,
        }

    def step(self, a):
        obs, _, done, info = self.base_env.step(a)
        reward = self.reward_fn(obs, self.goal)
        self.count += 1
        next_obs = {
            # add timestep
            'observation': np.r_[obs.copy(), self.count],
            'achieved_goal': obs[:2],
            'desired_goal': self.goal,
        }
        return next_obs, reward, done or self.count >= 500, info

    def render(self):
        self.base_env.render()

    def get_image(self):
        self.render()
        data = self.base_env.viewer.get_image()

        img_data = data[0]
        width = data[1]
        height = data[2]

        tmp = np.fromstring(img_data, dtype=np.uint8)
        image_obs = np.reshape(tmp, [height, width, 3])
        image_obs = np.flipud(image_obs)

        return image_obs

    @property
    def action_space(self):
        return self.base_env.action_space

    @property
    def observation_space(self):
        return self.base_env.observation_space

def run_environment(env_name, episode_length, num_episodes):
    env = EnvWithGoal(
            create_maze_env.create_maze_env(env_name),
            env_name)

    def action_fn(obs):
        action_space = env.action_space
        action_space_mean = (action_space.low + action_space.high) / 2.0
        action_space_magn = (action_space.high - action_space.low) / 2.0
        random_action = (action_space_mean +
            action_space_magn *
            np.random.uniform(low=-1.0, high=1.0,
            size=action_space.shape))

        return random_action

    rewards = []
    successes = []
    for ep in range(num_episodes):
        rewards.append(0.0)
        successes.append(False)
        obs = env.reset()
        for _ in range(episode_length):
            env.render()
            print(env.get_image().shape)
            obs, reward, done, _ = env.step(action_fn(obs))
            rewards[-1] += reward
            successes[-1] = success_fn(reward)
            if done:
                break
        
        print('Episode {} reward: {}, Success: {}'.format(ep + 1, rewards[-1], successes[-1]))

    print('Average Reward over {} episodes: {}'.format(num_episodes, np.mean(rewards)))
    print('Average Success over {} episodes: {}'.format(num_episodes, np.mean(successes)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="AntEnv", type=str)               
    parser.add_argument("--episode_length", default=500, type=int)      
    parser.add_argument("--num_episodes", default=100, type=int)

    args = parser.parse_args()
    run_environment(args.env_name, args.episode_length, args.num_episodes)