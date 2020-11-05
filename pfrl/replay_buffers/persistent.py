import os
import warnings

from pfrl.collections.persistent_collections import PersistentRandomAccessQueue

from .episodic import EpisodicReplayBuffer
from .replay_buffer import ReplayBuffer


class PersistentReplayBuffer(ReplayBuffer):
    """Experience replay buffer that are saved to disk storage

    :py:class:`ReplayBuffer` is used to store sampled experience data, but
    the data is stored in DRAM memory and removed after program termination.
    This class add persistence to :py:class:`ReplayBuffer`,
    so that the learning process can be restarted from a previously saved replay
    data.

    Args:
        dirname (str): Directory name where the buffer data is saved.
            Please note that it tries to load data from it as well. Also, it
            would be important to note that it can't be used with ancestor.
        capacity (int): Capacity in terms of number of transitions
        ancestor (str): Path to pre-generated replay buffer. The `ancestor`
            directory is used to load/save, instead of `dirname`.
        logger: logger object
        distributed (bool): Use a distributed version for the underlying
            persistent queue class. You need the private package `pfrlmn`
            to use this option.
        group: `torch.distributed` group object. Only used when
            `distributed=True` and pfrlmn package is available

    .. note:: Contrary to the original :py:class:`ReplayBuffer`
            implementation, ``state`` and ``next_state``, ``action`` and
            ``next_action`` are pickled and stored as different objects even
            they point to the same object. This may lead to inefficient usage
            of storage space, but it is recommended to buy more
            storage - hardware is sometimes cheaper than software.

    """

    def __init__(
        self,
        dirname,
        capacity,
        *,
        ancestor=None,
        logger=None,
        distributed=False,
        group=None
    ):
        super().__init__(capacity)

        if not distributed:
            self.memory = PersistentRandomAccessQueue(
                dirname, capacity, ancestor=ancestor, logger=logger
            )
        else:
            try:
                # Use distributed versions of PersistentRandomAccessQueue
                import pfrlmn.collections.persistent_collections as mn_coll

                self.memory = mn_coll.PersistentRandomAccessQueue(
                    dirname, capacity, ancestor=ancestor, logger=logger, group=group
                )

            except ImportError:
                # "pfrlmn" package is not publicly available as of pfrl release.
                raise RuntimeError(
                    "`pfrlmn` private package is required "
                    "to enable distributed execution support "
                    "of PersistentReplayBuffer."
                )

    def save(self, _):
        pass

    def load(self, _):
        warnings.warn(
            "{}.load() has been ignored, as it is persistent replay buffer".format(self)
        )


class PersistentEpisodicReplayBuffer(EpisodicReplayBuffer):
    """Episodic version of :py:class:`PersistentReplayBuffer`

    Args:
        dirname (str): Directory name where the buffer data is saved.
            This cannot be used with `ancestor`
        capacity (int): Capacity in terms of number of transitions
        ancestor (str): Path to pre-generated replay buffer. The `ancestor`
            directory is used to load/save, instead of `dirname`.
        logger: logger object
        distributed (bool): Use a distributed version for the underlying
            persistent queue class. You need the private package `pfrlmn`
            to use this option.
        group: `torch.distributed` group object. Only used when
            `distributed=True` and pfrlmn package is available

    .. note:: Current implementation is inefficient, as episodic
           memory and memory data shares the almost same data in
           :py:class:`EpisodicReplayBuffer` by reference but shows different
           data structure. Otherwise, persistent version of them does
           not share the data between them but backing file structure
           is separated.

    """

    def __init__(
        self,
        dirname,
        capacity,
        *,
        ancestor=None,
        logger=None,
        distributed=False,
        group=None
    ):
        super().__init__(capacity)

        self.memory_dir = os.path.join(dirname, "memory")
        self.episodic_memory_dir = os.path.join(dirname, "episodic_memory")

        if not distributed:
            self.memory = PersistentRandomAccessQueue(
                self.memory_dir, capacity, ancestor=ancestor, logger=logger
            )
            self.episodic_memory = PersistentRandomAccessQueue(
                self.episodic_memory_dir, capacity, ancestor=ancestor, logger=logger
            )
        else:
            try:
                # Use distributed versions of PersistentRandomAccessQueue
                import pfrlmn.collections.persistent_collections as mn_coll

                self.memory = mn_coll.PersistentRandomAccessQueue(
                    self.memory_dir,
                    capacity,
                    ancestor=ancestor,
                    logger=logger,
                    group=group,
                )
                self.episodic_memory = mn_coll.PersistentRandomAccessQueue(
                    self.episodic_memory_dir,
                    capacity,
                    ancestor=ancestor,
                    logger=logger,
                    group=group,
                )

            except ImportError:
                # "pfrlmn" package is not publicly available as of pfrl release.
                raise RuntimeError(
                    "`pfrlmn` private package is required "
                    "to enable distributed execution support "
                    "of PersistentEpisodicReplayBuffer."
                )

    def save(self, _):
        pass

    def load(self, _):
        warnings.warn(
            "PersistentEpisodicReplayBuffer.load() is called but it has not effect."
        )
