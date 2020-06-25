import tempfile

import pytest

from pfrl.collections.persistent_collections import PersistentRandomAccessQueue


def check_basic(tmpd):
    rb = PersistentRandomAccessQueue(tmpd, 16)
    assert 16 == rb.maxlen

    data = {0x42: "pocketburger"}
    rb.append(data)
    x = rb.sample(1)
    assert [data] == x
    assert data == x[0]
    assert 1 == len(rb)
    del x

    rb.popleft()
    assert 0 == len(rb)

    deadbeefs = []
    for i in range(10):
        assert i == len(rb)
        data = (0x42 + i, "deadbeef")
        rb.append(data)
        deadbeefs.append(data)

    assert 10 == len(rb)
    samples = rb.sample(10)
    assert sorted(deadbeefs) == sorted(samples)

    for i in range(10):
        data = (0x52 + i, "deadbeef")
        rb.append(data)
        deadbeefs.append(data)

    assert 16 == len(rb)
    deadbeefs = deadbeefs[-16:]
    samples = rb.sample(16)
    assert samples != sorted(samples)
    assert deadbeefs == sorted(samples)


def test_basic_single_node():
    with tempfile.TemporaryDirectory() as tmpd:
        check_basic(tmpd)


def test_recovery():
    deadbeefs = []
    with tempfile.TemporaryDirectory() as tmpd:
        for x in range(42):
            rb = PersistentRandomAccessQueue(tmpd, 16)
            assert 16 == rb.maxlen

            deadbeefs = deadbeefs[-16:]
            assert len(deadbeefs) == len(rb)

            samples = rb.sample(len(deadbeefs))
            if deadbeefs:
                assert samples != sorted(samples)
                assert deadbeefs == sorted(samples)

            for i in range(121):
                data = (0x42 + i + x, "deadbeef")
                rb.append(data)
                deadbeefs.append(data)

            rb.close()


@pytest.mark.parametrize(
    "maxlen,ancestors_level,datasizes",
    [
        (16, 2, (18, 0)),
        (7, 2, (13, 0)),
        (1, 2, (1, 0)),
        (1024, 2, (1024, 0)),
        (1024, 2, (18, 0)),
        (1024, 3, (18, 10, 0)),
        (1024, 3, (18, 0, 0)),
        (17, 3, (5, 7, 20)),
    ],
)
def test_ancestor(maxlen, ancestors_level, datasizes):
    # Test multiple depth of ancestors
    # maxlen: max length of the buffer(s)
    # ancestors_level: The number of ancestors to use
    # datasizes: data sizes to append() to each buffer
    assert len(datasizes) == ancestors_level
    c0bebeefs = []
    buffers = []
    tmpdirs = []

    for level in range(ancestors_level):
        datasize = datasizes[level]
        tmp_dir = tempfile.TemporaryDirectory()
        tmpdirs.append(tmp_dir)

        if level == 0:
            anc = None
        else:
            anc = tmpdirs[level - 1].name
        buf = PersistentRandomAccessQueue(tmp_dir.name, maxlen, ancestor=anc)
        buffers.append(buf)

        for i in range(datasize):
            data = (0x42 + i, "c0bebeef")
            buf.append(data)
            c0bebeefs.append(data)

        c0bebeefs = c0bebeefs[-maxlen:]
        assert len(c0bebeefs) == len(buf)

        buf.close()

    for d in tmpdirs:
        d.cleanup()
