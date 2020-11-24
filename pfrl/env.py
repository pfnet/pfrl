from abc import ABCMeta, abstractmethod


class Env(object, metaclass=ABCMeta):
    """RL learning environment.

    This serves a minimal interface for RL agents.
    """

    @abstractmethod
    def step(self, action):
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abstractmethod
    def close(self):
        raise NotImplementedError()


class VectorEnv(object, metaclass=ABCMeta):
    """Parallel RL learning environments."""

    @abstractmethod
    def step(self, action):
        raise NotImplementedError()

    @abstractmethod
    def reset(self, mask):
        """Reset envs.

        Args:
            mask (Sequence of bool): Mask array that specifies which env to
                skip. If omitted, all the envs are reset.
        """
        raise NotImplementedError()

    @abstractmethod
    def seed(self, seeds):
        raise NotImplementedError()

    @abstractmethod
    def close(self):
        raise NotImplementedError()

    @property
    def unwrapped(self):
        """Completely unwrap this env.

        Returns:
            VectorEnv: The base non-wrapped VectorEnv instance
        """
        return self
