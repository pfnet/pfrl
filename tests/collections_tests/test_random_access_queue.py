import collections

import pytest

from pfrl.collections.random_access_queue import RandomAccessQueue


@pytest.mark.parametrize("maxlen", [1, 10, None])
@pytest.mark.parametrize("init_seq", [None, [], range(5)])
class TestRandomAccessQueue:
    @pytest.fixture(autouse=True)
    def setUp(self, maxlen, init_seq):
        self.maxlen = maxlen
        self.init_seq = init_seq
        if self.init_seq:
            self.y_queue = RandomAccessQueue(self.init_seq, maxlen=self.maxlen)
            self.t_queue = collections.deque(self.init_seq, maxlen=self.maxlen)
        else:
            self.y_queue = RandomAccessQueue(maxlen=self.maxlen)
            self.t_queue = collections.deque(maxlen=self.maxlen)

    def test1(self):
        self.check_all()

        self.check_popleft()
        self.do_append(10)
        self.check_all()

        self.check_popleft()
        self.check_popleft()
        self.do_append(11)
        self.check_all()

        # test negative indices
        n = len(self.t_queue)
        for i in range(-n, 0):
            self.check_getitem(i)

        for k in range(4):
            self.do_extend(range(k))
            self.check_all()

        for k in range(4):
            self.check_popleft()
            self.do_extend(range(k))
            self.check_all()

        for k in range(10):
            self.do_append(20 + k)
            self.check_popleft()
            self.check_popleft()
            self.check_all()

        for _ in range(100):
            self.check_popleft()

    def check_all(self):
        self.check_len()
        n = len(self.t_queue)
        for i in range(n):
            self.check_getitem(i)

    def check_len(self):
        assert len(self.y_queue) == len(self.t_queue)

    def check_getitem(self, i):
        assert self.y_queue[i] == self.t_queue[i]

    def do_setitem(self, i, x):
        self.y_queue[i] = x
        self.t_queue[i] = x

    def do_append(self, x):
        self.y_queue.append(x)
        self.t_queue.append(x)

    def do_extend(self, xs):
        self.y_queue.extend(xs)
        self.t_queue.extend(xs)

    def check_popleft(self):
        try:
            t = self.t_queue.popleft()
        except IndexError:
            with pytest.raises(IndexError):
                self.y_queue.popleft()
        else:
            assert self.y_queue.popleft() == t
