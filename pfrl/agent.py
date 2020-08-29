from abc import ABCMeta
from abc import abstractmethod
from abc import abstractproperty
import contextlib
import os
from typing import Any
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import torch
import numpy as np
import time


class Agent(object, metaclass=ABCMeta):
    """Abstract agent class."""

    training = True

    @abstractmethod
    def act(self, obs: Any) -> Any:
        """Select an action.

        Returns:
            ~object: action
        """
        raise NotImplementedError()

    @abstractmethod
    def observe(self, obs: Any, reward: float, done: bool, reset: bool) -> None:
        """Observe consequences of the last action.

        Returns:
            None
        """
        raise NotImplementedError()

    @abstractmethod
    def save(self, dirname: str) -> None:
        """Save internal states.

        Returns:
            None
        """
        pass

    @abstractmethod
    def load(self, dirname: str) -> None:
        """Load internal states.

        Returns:
            None
        """
        pass

    @abstractmethod
    def get_statistics(self) -> List[Tuple[str, Any]]:
        """Get statistics of the agent.

        Returns:
            List of two-item tuples. The first item in a tuple is a str that
            represents the name of item, while the second item is a value to be
            recorded.

            Example: [('average_loss': 0), ('average_value': 1), ...]
        """
        pass

    @contextlib.contextmanager
    def eval_mode(self):
        orig_mode = self.training
        try:
            self.training = False
            yield
        finally:
            self.training = orig_mode


class HRLAgent(Agent, metaclass=ABCMeta):
    """Abstract HRL agent class."""
    training = True

    @abstractmethod
    def act_high_level(self, obs: Any, goal: Any):
        """
        high level controller act method
        """
        pass

    @abstractmethod
    def act_low_level(self, obs: Any, goal: Any):
        """
        low level controller act method
        """
        pass

    @abstractmethod
    def act(self, obs: Any, goal: Any):
        """
        Selects an action,
        based on a goal and observation.
        """
        pass

    def act(self, obs: Any) -> Any:
        """Select an action, based on an
        observation.

        Returns:
            ~object: action
        """
        raise NotImplementedError()


    def act(self, obs: Any, goal: Any) -> Any:
        """Select an action, based on a goal
        and observation.

        Returns:
            ~object: action
        """
        raise NotImplementedError()

    def observe(self, obs: Any, reward: float, done: bool, reset: bool) -> None:
        """Observe consequences of the last action.

        Returns:
            None
        """

    def set_final_goal(self, fg):
        """
        sets the final goal for the agent.
        """
        self.fg = fg

    def train(self, global_step: int) -> Any:
        """Trains the HRL agent.

        Returns:
            None
        """
        raise NotImplementedError()

    def append(self, step, s, a, n_s, r, d) -> Any:
        """
        Appends an experience to the replay buffer
        """
        raise NotImplementedError()

    def step(self, s, env, step, global_step=0, explore=False) -> Any:
        """
        Take an action and step in the environment.
        """
        raise NotImplementedError()

    def end_episode(self, episode, logger=None):
        raise NotImplementedError()

    def get_statistics(self) -> List[Tuple[str, Any]]:
        """Get statistics of the agent.

        Returns:
            List of two-item tuples. The first item in a tuple is a str that
            represents the name of item, while the second item is a value to be
            recorded.

            Example: [('average_loss': 0), ('average_value': 1), ...]
        """
        pass

    def evaluate_policy(self, env, eval_episodes=10, render=False, save_video=False, sleep=-1):
        if save_video:
            from OpenGL import GL
            import gym
            env = gym.wrappers.Monitor(env, directory='video',
                                       write_upon_reset=True, force=True, resume=True, mode='evaluation')
            render = False

        success = 0
        rewards = []
        env.evaluate = True
        for e in range(eval_episodes):
            obs = env.reset()
            fg = obs['desired_goal']
            s = obs['observation']
            done = False
            reward_episode_sum = 0
            step = 0

            self.set_final_goal(fg)

            while not done:
                if render:
                    env.render()
                if sleep > 0:
                    time.sleep(sleep)

                a, r, n_s, done = self.step(s, env, step)
                reward_episode_sum += r

                s = n_s
                step += 1
                self.end_step()
            else:
                error = np.sqrt(np.sum(np.square(fg-s[:2])))
                print('Goal, Curr: (%02.2f, %02.2f, %02.2f, %02.2f)     Error:%.2f' % (fg[0], fg[1], s[0], s[1], error))
                rewards.append(reward_episode_sum)
                success += 1 if error <= 5 else 0
                self.end_episode(e)

        env.evaluate = False
        return np.array(rewards), success/eval_episodes


class AttributeSavingMixin(object):
    """Mixin that provides save and load functionalities."""

    @abstractproperty
    def saved_attributes(self) -> Tuple[str, ...]:
        """Specify attribute names to save or load as a tuple of str."""
        pass

    def save(self, dirname: str) -> None:
        """Save internal states."""
        self.__save(dirname, [])

    def __save(self, dirname: str, ancestors: List[Any]):
        os.makedirs(dirname, exist_ok=True)
        ancestors.append(self)
        for attr in self.saved_attributes:
            assert hasattr(self, attr)
            attr_value = getattr(self, attr)
            if attr_value is None:
                continue
            if isinstance(attr_value, AttributeSavingMixin):
                assert not any(
                    attr_value is ancestor for ancestor in ancestors
                ), "Avoid an infinite loop"
                attr_value.__save(os.path.join(dirname, attr), ancestors)
            else:
                torch.save(
                    attr_value.state_dict(), os.path.join(dirname, "{}.pt".format(attr))
                )
        ancestors.pop()

    def load(self, dirname: str) -> None:
        """Load internal states."""
        self.__load(dirname, [])

    def __load(self, dirname: str, ancestors: List[Any]) -> None:
        map_location = torch.device("cpu") if not torch.cuda.is_available() else None
        ancestors.append(self)
        for attr in self.saved_attributes:
            assert hasattr(self, attr)
            attr_value = getattr(self, attr)
            if attr_value is None:
                continue
            if isinstance(attr_value, AttributeSavingMixin):
                assert not any(
                    attr_value is ancestor for ancestor in ancestors
                ), "Avoid an infinite loop"
                attr_value.load(os.path.join(dirname, attr))
            else:
                attr_value.load_state_dict(
                    torch.load(
                        os.path.join(dirname, "{}.pt".format(attr)), map_location
                    )
                )
        ancestors.pop()


class AsyncAgent(Agent, metaclass=ABCMeta):
    """Abstract asynchronous agent class."""

    @abstractproperty
    def process_idx(self) -> Optional[int]:
        """Index of process as integer, 0 for the representative process.

        The returned value can be None if it is not assgined yet.
        """
        pass

    @abstractproperty
    def shared_attributes(self) -> Tuple[str, ...]:
        """Tuple of names of shared attributes."""
        pass


class BatchAgent(Agent, metaclass=ABCMeta):
    """Abstract agent class that can interact with a batch of envs."""

    def act(self, obs: Any) -> Any:
        return self.batch_act([obs])[0]

    def observe(self, obs: Any, reward: float, done: bool, reset: bool) -> None:
        self.batch_observe([obs], [reward], [done], [reset])

    @abstractmethod
    def batch_act(self, batch_obs: Sequence[Any]) -> Sequence[Any]:
        """Select a batch of actions.

        Args:
            batch_obs (Sequence of ~object): Observations.

        Returns:
            Sequence of ~object: Actions.
        """
        raise NotImplementedError()

    @abstractmethod
    def batch_observe(
        self,
        batch_obs: Sequence[Any],
        batch_reward: Sequence[float],
        batch_done: Sequence[bool],
        batch_reset: Sequence[bool],
    ) -> None:
        """Observe a batch of action consequences.

        Args:
            batch_obs (Sequence of ~object): Observations.
            batch_reward (Sequence of float): Rewards.
            batch_done (Sequence of boolean): Boolean values where True
                indicates the current state is terminal.
            batch_reset (Sequence of boolean): Boolean values where True
                indicates the current episode will be reset, even if the
                current state is not terminal.

        Returns:
            None
        """
        raise NotImplementedError()


class GoalConditionedBatchAgent(BatchAgent, metaclass=ABCMeta):
    """Abstract GOAL conditioned agent class that can interact with a batch of envs."""

    def act_with_goal(self, obs: Any, goal: Any) -> Any:
        return self.batch_act_with_goal([obs], [goal])[0]

    def observe_with_goal(self, obs: Any, goals: Any, reward: float, done: bool, reset: bool) -> None:
        self.batch_observe_with_goal([obs], [goals], [reward], [done], [reset])

    def observe_with_goal_state_action_arr(self, state_arr: Any, action_arr: Any, obs: Any, goals: Any, reward: float, done: bool, reset: bool) -> None:
        self.batch_observe_with_goal_state_action_arr(state_arr, action_arr, [obs], [goals], [reward], [done], [reset])


    @abstractmethod
    def batch_act_with_goal(self, batch_obs: Sequence[Any], batch_goal: Sequence[Any]) -> Sequence[Any]:
        """Select a batch of actions.

        Args:
            batch_obs (Sequence of ~object): Observations.
            batch_goal (Sequence of ~object): Goals.

        Returns:
            Sequence of ~object: Actions.
        """
        raise NotImplementedError()

    @abstractmethod
    def batch_observe_with_goal(
        self,
        batch_obs: Sequence[Any],
        batch_goal: Sequence[Any],
        batch_reward: Sequence[float],
        batch_done: Sequence[bool],
        batch_reset: Sequence[bool],
    ) -> None:
        """Observe a batch of action consequences.

        Args:
            batch_obs (Sequence of ~object): Observations.
            batch_goal (Sequence of ~object): Goals.
            batch_reward (Sequence of float): Rewards.
            batch_done (Sequence of boolean): Boolean values where True
                indicates the current state is terminal.
            batch_reset (Sequence of boolean): Boolean values where True
                indicates the current episode will be reset, even if the
                current state is not terminal.

        Returns:
            None
        """
        raise NotImplementedError()

    @abstractmethod
    def batch_observe_with_goal_state_action_arr(
        self,
        state_arr: Sequence[Any],
        action_arr: Sequence[Any],
        batch_obs: Sequence[Any],
        batch_goal: Sequence[Any],
        batch_reward: Sequence[float],
        batch_done: Sequence[bool],
        batch_reset: Sequence[bool],
    ) -> None:
        """Observe a batch of action consequences.

        Args:
            state_arr (Sequence of a object): Recent states.
            action_arr (Sequence of a object): Recent actions.
            batch_obs (Sequence of object): Observations.
            batch_goal (Sequence of object): Goals.
            batch_reward (Sequence of float): Rewards.
            batch_done (Sequence of boolean): Boolean values where True
                indicates the current state is terminal.
            batch_reset (Sequence of boolean): Boolean values where True
                indicates the current episode will be reset, even if the
                current state is not terminal.

        Returns:
            None
        """
        raise NotImplementedError() 
