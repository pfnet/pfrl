from unittest import mock

import pytest

import pfrl


@pytest.mark.parametrize("max_episode_steps", [1, 2, 3])
def test_continuing_time_limit(max_episode_steps):
    env = mock.Mock()
    env.reset.side_effect = ["state"] * 2
    # Since info dicts are modified by the wapper, each step call needs to
    # return a new info dict.
    env.step.side_effect = [("state", 0, False, {}) for _ in range(6)]
    env = pfrl.wrappers.ContinuingTimeLimit(env, max_episode_steps=max_episode_steps)

    env.reset()
    for t in range(2):
        _, _, done, info = env.step(0)
        if t + 1 >= max_episode_steps:
            assert info["needs_reset"]
        else:
            assert not info.get("needs_reset", False)

    env.reset()
    for t in range(4):
        _, _, done, info = env.step(0)
        if t + 1 >= max_episode_steps:
            assert info["needs_reset"]
        else:
            assert not info.get("needs_reset", False)
