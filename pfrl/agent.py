import contextlib
import os
from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Any, List, Optional, Sequence, Tuple

import torch


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

            Example: [('average_loss', 0), ('average_value', 1), ...]
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
                if isinstance(
                    attr_value,
                    (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel),
                ):
                    attr_value = attr_value.module
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
                if isinstance(
                    attr_value,
                    (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel),
                ):
                    attr_value = attr_value.module
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
