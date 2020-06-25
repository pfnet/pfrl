from torch import nn

import pfrl


def test_evaluating():
    a = nn.Linear(1, 1)
    b = nn.Linear(1, 1)
    assert a.training
    assert b.training
    with pfrl.utils.evaluating(a):
        assert not a.training
        assert b.training
        with pfrl.utils.evaluating(b):
            assert not a.training
            assert not b.training
        assert not a.training
        assert b.training
    assert a.training
    assert b.training
