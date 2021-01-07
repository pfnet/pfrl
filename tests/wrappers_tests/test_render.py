from unittest import mock

import pytest

import pfrl


@pytest.mark.parametrize(
    "render_kwargs",
    [
        {},
        {"mode": "human"},
        {"mode": "rgb_array"},
    ],
)
def test_render(render_kwargs):
    orig_env = mock.Mock()
    # Reaches the terminal state after five actions
    orig_env.reset.side_effect = [
        ("state", 0),
        ("state", 3),
    ]
    orig_env.step.side_effect = [
        (("state", 1), 0, False, {}),
        (("state", 2), 1, True, {}),
    ]
    env = pfrl.wrappers.Render(orig_env, **render_kwargs)

    # Not called env.render yet
    assert orig_env.render.call_count == 0

    obs = env.reset()
    assert obs == ("state", 0)

    # Called once
    assert orig_env.render.call_count == 1

    obs, reward, done, info = env.step(0)
    assert obs == ("state", 1)
    assert reward == 0
    assert not done
    assert info == {}

    # Called twice
    assert orig_env.render.call_count == 2

    obs, reward, done, info = env.step(0)
    assert obs == ("state", 2)
    assert reward == 1
    assert done
    assert info == {}

    # Called thrice
    assert orig_env.render.call_count == 3

    obs = env.reset()
    assert obs == ("state", 3)

    # Called four times
    assert orig_env.render.call_count == 4

    # All the calls should receive correct kwargs
    for call in orig_env.render.call_args_list:
        args, kwargs = call
        assert len(args) == 0
        assert kwargs == render_kwargs
