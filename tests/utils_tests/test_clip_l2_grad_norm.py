import timeit
from logging import getLogger

import numpy as np
import pytest
import torch
from torch import nn

from pfrl.utils import clip_l2_grad_norm_


def _get_grad_vector(model):
    return np.concatenate(
        [p.grad.cpu().numpy().ravel().copy() for p in model.parameters()]
    )


def _test_clip_l2_grad_norm_(gpu):
    if gpu >= 0:
        device = torch.device("cuda:{}".format(gpu))
    else:
        device = torch.device("cpu")
    model = nn.Sequential(
        nn.Linear(2, 10),
        nn.ReLU(),
        nn.Linear(10, 3),
    ).to(device)
    x = torch.rand(7, 2).to(device)

    def backward():
        model.zero_grad()
        loss = model(x).mean()
        loss.backward()

    backward()
    raw_grads = _get_grad_vector(model)

    # Threshold large enough not to affect grads
    th = 10000
    backward()
    nn.utils.clip_grad_norm_(model.parameters(), th)
    clipped_grads = _get_grad_vector(model)

    backward()
    clip_l2_grad_norm_(model.parameters(), th)
    our_clipped_grads = _get_grad_vector(model)

    np.testing.assert_allclose(raw_grads, clipped_grads)
    np.testing.assert_allclose(raw_grads, our_clipped_grads)

    # Threshold small enough to affect grads
    th = 1e-2
    backward()
    nn.utils.clip_grad_norm_(model.parameters(), th)
    clipped_grads = _get_grad_vector(model)

    backward()
    clip_l2_grad_norm_(model.parameters(), th)
    our_clipped_grads = _get_grad_vector(model)

    with pytest.raises(AssertionError):
        np.testing.assert_allclose(raw_grads, clipped_grads, rtol=1e-5)

    with pytest.raises(AssertionError):
        np.testing.assert_allclose(raw_grads, our_clipped_grads, rtol=1e-5)

    np.testing.assert_allclose(clipped_grads, our_clipped_grads, rtol=1e-5)


def test_clip_l2_grad_norm_cpu():
    _test_clip_l2_grad_norm_(-1)


@pytest.mark.gpu
def test_clip_l2_grad_norm_gpu():
    _test_clip_l2_grad_norm_(0)


@pytest.mark.slow
def test_clip_l2_grad_norm_speed():
    logger = getLogger(__name__)

    # Speed difference is large when model is large
    model = nn.Sequential(
        nn.Linear(2, 1000),
        nn.ReLU(),
        nn.Linear(1000, 1000),
        nn.ReLU(),
        nn.Linear(1000, 3),
    )
    x = torch.rand(7, 2)

    def backward():
        model.zero_grad()
        loss = model(x).mean()
        loss.backward()

    # Threshold large enough not to affect grads
    th = 10000
    backward()

    def torch_clip():
        nn.utils.clip_grad_norm_(model.parameters(), th)

    torch_time = timeit.timeit(torch_clip, number=100)
    logger.debug("torch.nn.utils.clip_grad_norm_ took %s", torch_time)

    def our_clip():
        clip_l2_grad_norm_(model.parameters(), th)

    our_time = timeit.timeit(our_clip, number=100)
    logger.debug("pfrl.misc.clip_l2_grad_norm_ took %s", our_time)

    assert our_time < torch_time
