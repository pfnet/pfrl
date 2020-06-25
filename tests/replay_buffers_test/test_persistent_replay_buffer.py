import os
import tempfile

import pytest

from pfrl.replay_buffers import PersistentEpisodicReplayBuffer


@pytest.mark.parametrize("capacity", [None, 100])
class TestEpisodicReplayBuffer(object):
    def setup_method(self, method):
        self.tempdir = tempfile.TemporaryDirectory()

    def teardown_method(self, method):
        self.tempdir.cleanup()

    def test_append_and_sample(self, capacity):
        rbuf = PersistentEpisodicReplayBuffer(self.tempdir.name, capacity)
        assert len(rbuf) == 0

        for n in [10, 15, 5] * 3:
            transs = [
                dict(
                    state=i,
                    action=100 + i,
                    reward=200 + i,
                    next_state=i + 1,
                    next_action=101 + i,
                    is_state_terminal=(i == n - 1),
                )
                for i in range(n)
            ]
            for trans in transs:
                rbuf.append(**trans)

        assert len(rbuf) == 90
        assert rbuf.n_episodes == 9

        for k in [10, 30, 90]:
            s = rbuf.sample(k)
            assert len(s) == k

        for k in [1, 3, 9]:
            s = rbuf.sample_episodes(k)
            assert len(s) == k

            s = rbuf.sample_episodes(k, max_len=10)
            for ep in s:
                assert len(ep) <= 10
                for t0, t1 in zip(ep, ep[1:]):
                    assert t0["next_state"] == t1["state"]
                    assert t0["next_action"] == t1["action"]

    def test_save_and_load(self, capacity):
        tempdir = tempfile.mkdtemp()

        rbuf = PersistentEpisodicReplayBuffer(self.tempdir.name, capacity)

        transs = [
            dict(
                state=n,
                action=n + 10,
                reward=n + 20,
                next_state=n + 1,
                next_action=n + 11,
                is_state_terminal=False,
            )
            for n in range(5)
        ]

        # Add two episodes
        rbuf.append(**transs[0])
        rbuf.append(**transs[1])
        rbuf.stop_current_episode()

        rbuf.append(**transs[2])
        rbuf.append(**transs[3])
        rbuf.append(**transs[4])
        rbuf.stop_current_episode()

        assert len(rbuf) == 5
        assert rbuf.n_episodes == 2

        # Save
        filename = os.path.join(tempdir, "rbuf.pkl")
        # Actually does nothing
        rbuf.save(filename)
        del rbuf

        # Re-initialize rbuf
        rbuf = PersistentEpisodicReplayBuffer(self.tempdir.name, capacity)

        # Sampled transitions are exactly what I added!
        s5 = rbuf.sample(5)
        assert len(s5) == 5
        for t in s5:
            assert len(t) == 1
            n = t[0]["state"]
            assert n in range(5)
            assert t[0] == transs[n]

        # And sampled episodes are exactly what I added!
        s2e = rbuf.sample_episodes(2)
        assert len(s2e) == 2
        if s2e[0][0]["state"] == 0:
            assert s2e[0] == [transs[0], transs[1]]
            assert s2e[1] == [transs[2], transs[3], transs[4]]
        else:
            assert s2e[0] == [transs[2], transs[3], transs[4]]
            assert s2e[1] == [transs[0], transs[1]]

        # Sizes are correct!
        assert len(rbuf) == 5
        assert rbuf.n_episodes == 2


class TestEpisodicReplayBufferWithEnvID(object):
    def setup_method(self):
        self.tempdir = tempfile.TemporaryDirectory()

    def teardown_method(self):
        self.tempdir.cleanup()

    def test(self):
        rbuf = PersistentEpisodicReplayBuffer(self.tempdir.name, capacity=None)

        # 2 transitions for env_id=0
        for i in range(2):
            trans1 = dict(
                state=0,
                action=1,
                reward=2,
                next_state=3,
                next_action=4,
                is_state_terminal=False,
            )
            rbuf.append(env_id=0, **trans1)
        # 4 transitions for env_id=1 with a terminal state
        for i in range(4):
            trans1 = dict(
                state=0,
                action=1,
                reward=2,
                next_state=3,
                next_action=4,
                is_state_terminal=(i == 3),
            )
            rbuf.append(env_id=1, **trans1)
        # 9 transitions for env_id=2
        for i in range(9):
            trans1 = dict(
                state=0,
                action=1,
                reward=2,
                next_state=3,
                next_action=4,
                is_state_terminal=False,
            )
            rbuf.append(env_id=2, **trans1)

        # It should have 4 transitions from env_id=1
        assert len(rbuf) == 4

        # env_id=0 episode ends
        rbuf.stop_current_episode(env_id=0)

        # Now it should have 4 + 2 = 6 transitions
        assert len(rbuf) == 6

        # env_id=2 episode ends
        rbuf.stop_current_episode(env_id=2)

        # Finally it should have 4 + 2 + 9 = 15 transitions
        assert len(rbuf) == 15
