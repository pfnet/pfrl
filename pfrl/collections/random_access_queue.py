import itertools

from pfrl.utils.random import sample_n_k


class RandomAccessQueue(object):
    """FIFO queue with fast indexing

    Operations getitem, setitem, append, popleft, and len
    are amortized O(1)-time, if this data structure is used ephemerally.
    """

    def __init__(self, *args, **kwargs):
        self.maxlen = kwargs.pop("maxlen", None)
        assert self.maxlen is None or self.maxlen >= 0
        self._queue_front = []
        self._queue_back = list(*args, **kwargs)
        self._apply_maxlen()

    def _apply_maxlen(self):
        if self.maxlen is not None:
            while len(self) > self.maxlen:
                self.popleft()

    def __iter__(self):
        return itertools.chain(reversed(self._queue_front), iter(self._queue_back))

    def __repr__(self):
        return "RandomAccessQueue({})".format(str(list(iter(self))))

    def __len__(self):
        return len(self._queue_front) + len(self._queue_back)

    def __getitem__(self, i):
        if i >= 0:
            nf = len(self._queue_front)
            if i < nf:
                return self._queue_front[~i]
            else:
                i -= nf
                if i < len(self._queue_back):
                    return self._queue_back[i]
                else:
                    raise IndexError("RandomAccessQueue index out of range")

        else:
            nb = len(self._queue_back)
            if i >= -nb:
                return self._queue_back[i]
            else:
                i += nb
                if i >= -len(self._queue_front):
                    return self._queue_front[~i]
                else:
                    raise IndexError("RandomAccessQueue index out of range")

    def __setitem__(self, i, x):
        if i >= 0:
            nf = len(self._queue_front)
            if i < nf:
                self._queue_front[~i] = x
            else:
                i -= nf
                if i < len(self._queue_back):
                    self._queue_back[i] = x
                else:
                    raise IndexError("RandomAccessQueue index out of range")

        else:
            nb = len(self._queue_back)
            if i >= -nb:
                self._queue_back[i] = x
            else:
                i += nb
                if i >= -len(self._queue_front):
                    self._queue_front[~i] = x
                else:
                    raise IndexError("RandomAccessQueue index out of range")

    def append(self, x):
        self._queue_back.append(x)
        if self.maxlen is not None and len(self) > self.maxlen:
            self.popleft()

    def extend(self, xs):
        self._queue_back.extend(xs)
        self._apply_maxlen()

    def popleft(self):
        if not self._queue_front:
            if not self._queue_back:
                raise IndexError("pop from empty RandomAccessQueue")

            self._queue_front = self._queue_back
            self._queue_back = []
            self._queue_front.reverse()

        return self._queue_front.pop()

    def sample(self, k):
        return [self[i] for i in sample_n_k(len(self), k)]
