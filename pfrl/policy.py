from abc import ABCMeta, abstractmethod
from logging import getLogger

logger = getLogger(__name__)


class Policy(object, metaclass=ABCMeta):
    """Abstract policy."""

    @abstractmethod
    def __call__(self, state):
        """Evaluate a policy.

        Returns:
            Distribution of actions
        """
        raise NotImplementedError()
