import collections
from numbers import Number
from typing import (
    Any,
    Callable,
    Deque,
    Generic,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

import numpy as np

from pfrl.utils.random import sample_n_k

T = TypeVar("T")


class PrioritizedBuffer(Generic[T]):
    def __init__(
        self,
        capacity: Optional[int] = None,
        wait_priority_after_sampling: bool = True,
        initial_max_priority: float = 1.0,
    ):
        self.capacity = capacity
        self.data: Deque = collections.deque()
        self.priority_sums = SumTreeQueue()
        self.priority_mins = MinTreeQueue()
        self.max_priority = initial_max_priority
        self.wait_priority_after_sampling = wait_priority_after_sampling
        self.flag_wait_priority = False

    def __len__(self) -> int:
        return len(self.data)

    def append(self, value: T, priority: Optional[float] = None) -> None:
        if self.capacity is not None and len(self) == self.capacity:
            self.popleft()
        if priority is None:
            # Append with the highest priority
            priority = self.max_priority

        self.data.append(value)
        self.priority_sums.append(priority)
        self.priority_mins.append(priority)

    def popleft(self) -> T:
        assert len(self) > 0
        self.priority_sums.popleft()
        self.priority_mins.popleft()
        return self.data.popleft()

    def _sample_indices_and_probabilities(
        self, n: int, uniform_ratio: float
    ) -> Tuple[List[int], List[float], float]:
        total_priority: float = self.priority_sums.sum()
        min_prob = self.priority_mins.min() / total_priority
        indices = []
        priorities = []
        if uniform_ratio > 0:
            # Mix uniform samples and prioritized samples
            n_uniform = np.random.binomial(n, uniform_ratio)
            un_indices, un_priorities = self.priority_sums.uniform_sample(
                n_uniform, remove=self.wait_priority_after_sampling
            )
            indices.extend(un_indices)
            priorities.extend(un_priorities)
            n -= n_uniform
            min_prob = uniform_ratio / len(self) + (1 - uniform_ratio) * min_prob

        pr_indices, pr_priorities = self.priority_sums.prioritized_sample(
            n, remove=self.wait_priority_after_sampling
        )
        indices.extend(pr_indices)
        priorities.extend(pr_priorities)

        probs = [
            uniform_ratio / len(self) + (1 - uniform_ratio) * pri / total_priority
            for pri in priorities
        ]
        return indices, probs, min_prob

    def sample(
        self, n: int, uniform_ratio: float = 0
    ) -> Tuple[List[T], List[float], float]:
        """Sample data along with their corresponding probabilities.

        Args:
            n (int): Number of data to sample.
            uniform_ratio (float): Ratio of uniformly sampled data.
        Returns:
            sampled data (list)
            probabitilies (list)
        """
        assert not self.wait_priority_after_sampling or not self.flag_wait_priority
        indices, probabilities, min_prob = self._sample_indices_and_probabilities(
            n, uniform_ratio=uniform_ratio
        )
        sampled = [self.data[i] for i in indices]
        self.sampled_indices = indices
        self.flag_wait_priority = True
        return sampled, probabilities, min_prob

    def set_last_priority(self, priority: Sequence[float]) -> None:
        assert not self.wait_priority_after_sampling or self.flag_wait_priority
        assert all([p > 0.0 for p in priority])
        assert len(self.sampled_indices) == len(priority)
        for i, p in zip(self.sampled_indices, priority):
            self.priority_sums[i] = p
            self.priority_mins[i] = p
            self.max_priority = max(self.max_priority, p)
        self.flag_wait_priority = False
        self.sampled_indices = []

    def _uniform_sample_indices_and_probabilities(
        self, n: int
    ) -> Tuple[List[int], List[float]]:
        indices = list(sample_n_k(len(self.data), n))
        probabilities = [1 / len(self)] * len(indices)
        return indices, probabilities


# Implement operations on nodes of SumTreeQueue


# node = left_child, right_child, value
Node = List[Any]
V = TypeVar("V")
Aggregator = Callable[[Sequence[V]], V]


def _expand(node: Node) -> None:
    if not node:
        node[:] = [], [], None


def _reduce(node: Node, op: Aggregator) -> None:
    assert node
    left_node, right_node, _ = node
    parent_value = []
    if left_node:
        parent_value.append(left_node[2])
    if right_node:
        parent_value.append(right_node[2])
    if parent_value:
        node[2] = op(parent_value)
    else:
        del node[:]


def _write(
    index_left: int,
    index_right: int,
    node: Node,
    key: int,
    value: Optional[V],
    op: Aggregator,
) -> Optional[V]:
    if index_right - index_left == 1:
        if node:
            ret = node[2]
        else:
            ret = None
        if value is None:
            del node[:]
        else:
            node[:] = None, None, value
    else:
        _expand(node)
        node_left, node_right, _ = node
        index_center = (index_left + index_right) // 2
        if key < index_center:
            ret = _write(index_left, index_center, node_left, key, value, op)
        else:
            ret = _write(index_center, index_right, node_right, key, value, op)
        _reduce(node, op)
    return ret


class TreeQueue(Generic[V]):
    """Queue with Binary Indexed Tree cache

    queue-like data structure
    append, update are O(log n)
    reduction over an interval is O(log n) per query
    """

    root: Node
    bounds: Tuple[int, int]

    def __init__(self, op: Aggregator):
        self.length = 0
        self.op = op

    def __setitem__(self, ix: int, val: V) -> None:
        assert 0 <= ix < self.length
        assert val is not None
        self._write(ix, val)

    def _write(self, ix: int, val: Optional[V]) -> Optional[V]:
        ixl, ixr = self.bounds
        return _write(ixl, ixr, self.root, ix, val, self.op)

    def append(self, value: V) -> None:
        if self.length == 0:
            self.root = [None, None, value]
            self.bounds = 0, 1
            self.length = 1
            return

        ixl, ixr = self.bounds
        root = self.root
        if ixr == self.length:
            _, _, root_value = root
            self.root = [self.root, [], root_value]
            ixr += ixr - ixl
            self.bounds = ixl, ixr
        ret = self._write(self.length, value)
        assert ret is None
        self.length += 1

    def popleft(self) -> Optional[V]:
        assert self.length > 0
        ret = self._write(0, None)
        ixl, ixr = self.bounds
        ixl -= 1
        ixr -= 1
        self.length -= 1
        if self.length == 0:
            del self.root
            del self.bounds
            return ret

        ixc = (ixl + ixr) // 2
        if ixc == 0:
            ixl = ixc
            _, self.root, _ = self.root
        self.bounds = ixl, ixr
        return ret


def _find(index_left: int, index_right: int, node: Node, pos: Number) -> int:
    if index_right - index_left == 1:
        return index_left
    else:
        node_left, node_right, _ = node
        index_center = (index_left + index_right) // 2
        if node_left:
            left_value = node_left[2]
        else:
            left_value = 0.0
        if pos < left_value:
            return _find(index_left, index_center, node_left, pos)
        else:
            return _find(index_center, index_right, node_right, pos - left_value)


class SumTreeQueue(TreeQueue[float]):
    """Fast weighted sampling.

    queue-like data structure
    append, update are O(log n)
    summation over an interval is O(log n) per query
    """

    def __init__(self):
        super().__init__(op=sum)

    def sum(self) -> float:
        if self.length == 0:
            return 0.0
        else:
            return self.root[2]

    def uniform_sample(self, n: int, remove: bool) -> Tuple[List[int], List[float]]:
        assert n >= 0
        ixs = list(sample_n_k(self.length, n))
        vals: List[float] = []
        if n > 0:
            for ix in ixs:
                val = self._write(ix, 0.0)
                assert val is not None
                vals.append(val)

        if not remove:
            for ix, val in zip(ixs, vals):
                self._write(ix, val)

        return ixs, vals

    def prioritized_sample(self, n: int, remove: bool) -> Tuple[List[int], List[float]]:
        assert n >= 0
        ixs: List[int] = []
        vals: List[float] = []
        if n > 0:
            root = self.root
            ixl, ixr = self.bounds
            for _ in range(n):
                ix = _find(ixl, ixr, root, np.random.uniform(0.0, root[2]))
                val = self._write(ix, 0.0)
                assert val is not None
                ixs.append(ix)
                vals.append(val)

        if not remove:
            for ix, val in zip(ixs, vals):
                self._write(ix, val)

        return ixs, vals


class MinTreeQueue(TreeQueue[float]):
    def __init__(self):
        super().__init__(op=min)

    def min(self) -> float:
        if self.length == 0:
            return np.inf
        else:
            return self.root[2]
