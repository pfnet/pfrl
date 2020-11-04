import collections
import copy
import os
import tempfile
import unittest

import numpy as np
import pytest
import torch

from pfrl import replay_buffer, replay_buffers


@pytest.mark.parametrize("capacity", [100, None])
@pytest.mark.parametrize("num_steps", [1, 3])
class TestReplayBuffer:
    @pytest.fixture(autouse=True)
    def setUp(self, capacity, num_steps):
        self.capacity = capacity
        self.num_steps = num_steps

    def test_append_and_sample(self):
        capacity = self.capacity
        num_steps = self.num_steps
        rbuf = replay_buffers.ReplayBuffer(capacity, num_steps)

        assert len(rbuf) == 0

        # Add one and sample one
        correct_item = collections.deque([], maxlen=num_steps)
        for _ in range(num_steps):
            trans1 = dict(
                state=0,
                action=1,
                reward=2,
                next_state=3,
                next_action=4,
                is_state_terminal=False,
            )
            correct_item.append(trans1)
            rbuf.append(**trans1)
        assert len(rbuf) == 1
        s1 = rbuf.sample(1)
        assert len(s1) == 1
        assert s1[0] == list(correct_item)

        # Add two and sample two, which must be unique
        correct_item2 = copy.deepcopy(correct_item)
        trans2 = dict(
            state=1,
            action=1,
            reward=2,
            next_state=3,
            next_action=4,
            is_state_terminal=False,
        )
        correct_item2.append(trans2)
        rbuf.append(**trans2)
        assert len(rbuf) == 2
        s2 = rbuf.sample(2)
        assert len(s2) == 2
        if s2[0][num_steps - 1]["state"] == 0:
            assert s2[0] == list(correct_item)
            assert s2[1] == list(correct_item2)
        else:
            assert s2[1] == list(correct_item)
            assert s2[0] == list(correct_item2)

    def test_append_and_terminate(self):
        capacity = self.capacity
        num_steps = self.num_steps
        rbuf = replay_buffers.ReplayBuffer(capacity, num_steps)

        assert len(rbuf) == 0

        # Add one and sample one
        for _ in range(num_steps):
            trans1 = dict(
                state=0,
                action=1,
                reward=2,
                next_state=3,
                next_action=4,
                is_state_terminal=False,
            )
            rbuf.append(**trans1)
        assert len(rbuf) == 1
        s1 = rbuf.sample(1)
        assert len(s1) == 1

        # Add two and sample two, which must be unique
        trans2 = dict(
            state=1,
            action=1,
            reward=2,
            next_state=3,
            next_action=4,
            is_state_terminal=True,
        )
        rbuf.append(**trans2)
        assert len(rbuf) == self.num_steps + 1
        s2 = rbuf.sample(self.num_steps + 1)
        assert len(s2) == self.num_steps + 1
        if self.num_steps == 1:
            if s2[0][0]["state"] == 0:
                assert s2[1][0]["state"] == 1
            else:
                assert s2[1][0]["state"] == 0
        else:
            for item in s2:
                # e.g. if states are 0,0,0,1 then buffer looks like:
                # [[0,0,0], [0, 0, 1], [0, 1], [1]]
                if len(item) < self.num_steps:
                    assert item[len(item) - 1]["state"] == 1
                    for i in range(len(item) - 1):
                        assert item[i]["state"] == 0
                else:
                    for i in range(len(item) - 1):
                        assert item[i]["state"] == 0

    def test_stop_current_episode(self):
        capacity = self.capacity
        num_steps = self.num_steps
        rbuf = replay_buffers.ReplayBuffer(capacity, num_steps)

        assert len(rbuf) == 0

        # Add one and sample one
        for _ in range(num_steps - 1):
            trans1 = dict(
                state=0,
                action=1,
                reward=2,
                next_state=3,
                next_action=4,
                is_state_terminal=False,
            )
            rbuf.append(**trans1)
        # we haven't experienced n transitions yet
        assert len(rbuf) == 0
        # episode ends
        rbuf.stop_current_episode()
        # episode ends, so we should add n-1 transitions
        assert len(rbuf) == self.num_steps - 1

    def test_save_and_load(self):
        capacity = self.capacity
        num_steps = self.num_steps

        tempdir = tempfile.mkdtemp()

        rbuf = replay_buffers.ReplayBuffer(capacity, num_steps)

        correct_item = collections.deque([], maxlen=num_steps)
        # Add two transitions
        for _ in range(num_steps):
            trans1 = dict(
                state=0,
                action=1,
                reward=2,
                next_state=3,
                next_action=4,
                is_state_terminal=False,
            )
            correct_item.append(trans1)
            rbuf.append(**trans1)
        correct_item2 = copy.deepcopy(correct_item)
        trans2 = dict(
            state=1,
            action=1,
            reward=2,
            next_state=3,
            next_action=4,
            is_state_terminal=False,
        )
        correct_item2.append(trans2)
        rbuf.append(**trans2)

        # Now it has two transitions
        assert len(rbuf) == 2

        # Save
        filename = os.path.join(tempdir, "rbuf.pkl")
        rbuf.save(filename)

        # Initialize rbuf
        rbuf = replay_buffers.ReplayBuffer(capacity)

        # Of course it has no transition yet
        assert len(rbuf) == 0

        # Load the previously saved buffer
        rbuf.load(filename)

        # Now it has two transitions again
        assert len(rbuf) == 2

        # And sampled transitions are exactly what I added!
        s2 = rbuf.sample(2)
        if s2[0][num_steps - 1]["state"] == 0:
            assert s2[0] == list(correct_item)
            assert s2[1] == list(correct_item2)
        else:
            assert s2[0] == list(correct_item2)
            assert s2[1] == list(correct_item)


@pytest.mark.parametrize("capacity", [100, None])
class TestEpisodicReplayBuffer:
    @pytest.fixture(autouse=True)
    def setUp(self, capacity):
        self.capacity = capacity

    def test_append_and_sample(self):
        capacity = self.capacity
        rbuf = replay_buffers.EpisodicReplayBuffer(capacity)

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

    def test_save_and_load(self):
        capacity = self.capacity

        tempdir = tempfile.mkdtemp()

        rbuf = replay_buffers.EpisodicReplayBuffer(capacity)

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
        rbuf.save(filename)

        # Initialize rbuf
        rbuf = replay_buffers.EpisodicReplayBuffer(capacity)

        # Of course it has no transition yet
        assert len(rbuf) == 0

        # Load the previously saved buffer
        rbuf.load(filename)

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


@pytest.mark.parametrize("capacity", [100, None])
@pytest.mark.parametrize("normalize_by_max", ["batch", "memory"])
class TestPrioritizedReplayBuffer:
    @pytest.fixture(autouse=True)
    def setUp(self, capacity, normalize_by_max):
        self.capacity = capacity
        self.normalize_by_max = normalize_by_max
        self.num_steps = 1

    def test_append_and_sample(self):
        capacity = self.capacity
        num_steps = self.num_steps
        rbuf = replay_buffers.PrioritizedReplayBuffer(
            capacity,
            normalize_by_max=self.normalize_by_max,
            error_max=5,
            num_steps=num_steps,
        )

        assert len(rbuf) == 0

        # Add one and sample one
        correct_item = collections.deque([], maxlen=num_steps)
        for _ in range(num_steps):
            trans1 = dict(
                state=0,
                action=1,
                reward=2,
                next_state=3,
                next_action=4,
                is_state_terminal=False,
            )
            correct_item.append(trans1)
            rbuf.append(**trans1)
        assert len(rbuf) == 1
        s1 = rbuf.sample(1)
        rbuf.update_errors([3.14])
        assert len(s1) == 1
        np.testing.assert_allclose(s1[0][0]["weight"], 1.0)
        del s1[0][0]["weight"]
        assert s1[0] == list(correct_item)

        # Add two and sample two, which must be unique
        correct_item2 = copy.deepcopy(correct_item)
        trans2 = dict(
            state=1,
            action=1,
            reward=2,
            next_state=3,
            next_action=4,
            is_state_terminal=True,
        )
        correct_item2.append(trans2)
        rbuf.append(**trans2)
        assert len(rbuf) == 2
        s2 = rbuf.sample(2)
        rbuf.update_errors([3.14, 2.71])
        assert len(s2) == 2
        del s2[0][0]["weight"]
        del s2[1][0]["weight"]
        if s2[0][num_steps - 1]["state"] == 1:
            assert s2[0] == list(correct_item2)
            assert s2[1] == list(correct_item)
        else:
            assert s2[0] == list(correct_item)
            assert s2[1] == list(correct_item2)

        # Weights should be different for different TD-errors
        s3 = rbuf.sample(2)
        assert not np.allclose(s3[0][0]["weight"], s3[1][0]["weight"])

        # Weights should be equal for different but clipped TD-errors
        rbuf.update_errors([5, 10])
        s3 = rbuf.sample(2)
        np.testing.assert_allclose(s3[0][0]["weight"], s3[1][0]["weight"])

        # Weights should be equal for the same TD-errors
        rbuf.update_errors([3.14, 3.14])
        s4 = rbuf.sample(2)
        np.testing.assert_allclose(s4[0][0]["weight"], s4[1][0]["weight"])

    def test_normalize_by_max(self):

        rbuf = replay_buffers.PrioritizedReplayBuffer(
            self.capacity,
            normalize_by_max=self.normalize_by_max,
            error_max=1000,
            num_steps=self.num_steps,
        )

        # Add 100 transitions
        for i in range(100):
            trans = dict(
                state=i,
                action=1,
                reward=2,
                next_state=i + 1,
                next_action=1,
                is_state_terminal=False,
            )
            rbuf.append(**trans)
        assert len(rbuf) == 100

        def set_errors_based_on_state(rbuf, samples):
            # Use the value of 'state' as an error, so that state 0 will have
            # the smallest error, thus the largest weight
            errors = [s[0]["state"] for s in samples]
            rbuf.update_errors(errors)

        # Assign different errors to all the transitions first
        samples = rbuf.sample(100)
        set_errors_based_on_state(rbuf, samples)

        # Repeatedly check how weights are normalized
        for i in range(100):
            samples = rbuf.sample(i + 1)
            # All the weights must be unique
            assert len(set(s[0]["weight"] for s in samples)) == len(samples)
            # Now check the maximum weight in a minibatch
            max_w = max([s[0]["weight"] for s in samples])
            if self.normalize_by_max == "batch":
                # Maximum weight in a minibatch must be 1
                np.testing.assert_allclose(max_w, 1)
            elif self.normalize_by_max == "memory":
                # Maximum weight in a minibatch must be less than 1 unless
                # the minibatch contains the transition of least error.
                if any(s[0]["state"] == 0 for s in samples):
                    np.testing.assert_allclose(max_w, 1)
                else:
                    assert max_w < 1
            set_errors_based_on_state(rbuf, samples)

    def test_capacity(self):
        capacity = self.capacity
        if capacity is None:
            return

        rbuf = replay_buffers.PrioritizedReplayBuffer(capacity)
        # Fill the buffer
        for _ in range(capacity):
            trans1 = dict(
                state=0,
                action=1,
                reward=2,
                next_state=3,
                next_action=4,
                is_state_terminal=True,
            )
            rbuf.append(**trans1)
        assert len(rbuf) == capacity

        # Add a new transition
        trans2 = dict(
            state=1,
            action=1,
            reward=2,
            next_state=3,
            next_action=4,
            is_state_terminal=True,
        )
        rbuf.append(**trans2)
        # The size should not change
        assert len(rbuf) == capacity

    def test_save_and_load(self):
        capacity = self.capacity
        num_steps = self.num_steps

        tempdir = tempfile.mkdtemp()

        rbuf = replay_buffers.PrioritizedReplayBuffer(capacity, num_steps=num_steps)

        # Add two transitions
        correct_item = collections.deque([], maxlen=num_steps)
        for _ in range(num_steps):
            trans1 = dict(
                state=0,
                action=1,
                reward=2,
                next_state=3,
                next_action=4,
                is_state_terminal=False,
            )
            correct_item.append(trans1)
            rbuf.append(**trans1)
        correct_item2 = copy.deepcopy(correct_item)
        trans2 = dict(
            state=1,
            action=1,
            reward=2,
            next_state=3,
            next_action=4,
            is_state_terminal=True,
        )
        correct_item2.append(trans2)
        rbuf.append(**trans2)

        # Now it has two transitions
        assert len(rbuf) == 2

        # Save
        filename = os.path.join(tempdir, "rbuf.pkl")
        rbuf.save(filename)

        # Initialize rbuf
        rbuf = replay_buffers.PrioritizedReplayBuffer(capacity, num_steps=num_steps)

        # Of course it has no transition yet
        assert len(rbuf) == 0

        # Load the previously saved buffer
        rbuf.load(filename)

        # Now it has two transitions again
        assert len(rbuf) == 2

        # And sampled transitions are exactly what I added!
        s2 = rbuf.sample(2)
        del s2[0][0]["weight"]
        del s2[1][0]["weight"]
        if s2[0][num_steps - 1]["state"] == 0:
            assert s2[0] == list(correct_item)
            assert s2[1] == list(correct_item2)
        else:
            assert s2[0] == list(correct_item2)
            assert s2[1] == list(correct_item)


def exp_return_of_episode(episode):
    return sum(np.exp(x["reward"]) for x in episode)


@pytest.mark.parametrize("normalize_by_max", ["batch", "memory"])
@pytest.mark.parametrize(
    "wait_priority_after_sampling,default_priority_func",
    [(True, None), (True, exp_return_of_episode), (False, exp_return_of_episode)],
)
@pytest.mark.parametrize("uniform_ratio", [0, 0.1, 1.0])
@pytest.mark.parametrize("return_sample_weights", [True, False])
class TestPrioritizedEpisodicReplayBuffer:
    @pytest.fixture(autouse=True)
    def setUp(
        self,
        normalize_by_max,
        wait_priority_after_sampling,
        default_priority_func,
        uniform_ratio,
        return_sample_weights,
    ):
        self.capacity = 100
        self.normalize_by_max = normalize_by_max
        self.wait_priority_after_sampling = wait_priority_after_sampling
        self.default_priority_func = default_priority_func
        self.uniform_ratio = uniform_ratio
        self.return_sample_weights = return_sample_weights

    def test_append_and_sample(self):
        rbuf = replay_buffers.PrioritizedEpisodicReplayBuffer(
            capacity=self.capacity,
            normalize_by_max=self.normalize_by_max,
            default_priority_func=self.default_priority_func,
            uniform_ratio=self.uniform_ratio,
            wait_priority_after_sampling=self.wait_priority_after_sampling,
            return_sample_weights=self.return_sample_weights,
        )

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
            ret = rbuf.sample_episodes(k)
            if self.return_sample_weights:
                s, wt = ret
                assert len(s) == k
                assert len(wt) == k
            else:
                s = ret
                assert len(s) == k
            if self.wait_priority_after_sampling:
                rbuf.update_errors([1.0] * k)

            ret = rbuf.sample_episodes(k, max_len=10)
            if self.return_sample_weights:
                s, wt = ret
                assert len(s) == k
                assert len(wt) == k
            else:
                s = ret
            if self.wait_priority_after_sampling:
                rbuf.update_errors([1.0] * k)

            for ep in s:
                assert len(ep) <= 10
                for t0, t1 in zip(ep, ep[1:]):
                    assert t0["next_state"] == t1["state"]
                    assert t0["next_action"] == t1["action"]


@pytest.mark.parametrize(
    "replay_buffer_type", ["ReplayBuffer", "PrioritizedReplayBuffer"]
)
class TestReplayBufferWithEnvID:
    @pytest.fixture(autouse=True)
    def setUp(self, replay_buffer_type):
        self.replay_buffer_type = replay_buffer_type

    def test(self):
        n = 5
        if self.replay_buffer_type == "ReplayBuffer":
            rbuf = replay_buffers.ReplayBuffer(capacity=None, num_steps=n)
        elif self.replay_buffer_type == "PrioritizedReplayBuffer":
            rbuf = replay_buffers.PrioritizedReplayBuffer(capacity=None, num_steps=n)
        else:
            assert False

        # 2 transitions for env_id=0
        for _ in range(2):
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
        for _ in range(9):
            trans1 = dict(
                state=0,
                action=1,
                reward=2,
                next_state=3,
                next_action=4,
                is_state_terminal=False,
            )
            rbuf.append(env_id=2, **trans1)

        # It should have:
        #   - 4 transitions from env_id=1
        #   - 5 transitions from env_id=2
        assert len(rbuf) == 9

        # env_id=0 episode ends
        rbuf.stop_current_episode(env_id=0)

        # Now it should have 9 + 2 = 11 transitions
        assert len(rbuf) == 11

        # env_id=2 episode ends
        rbuf.stop_current_episode(env_id=2)

        # Finally it should have 9 + 2 + 4 = 15 transitions
        assert len(rbuf) == 15


@pytest.mark.parametrize(
    "replay_buffer_type", ["EpisodicReplayBuffer", "PrioritizedEpisodicReplayBuffer"]
)
class TestEpisodicReplayBufferWithEnvID:
    @pytest.fixture(autouse=True)
    def setUp(self, replay_buffer_type):
        self.replay_buffer_type = replay_buffer_type

    def test(self):
        if self.replay_buffer_type == "EpisodicReplayBuffer":
            rbuf = replay_buffers.EpisodicReplayBuffer(capacity=None)
        elif self.replay_buffer_type == "PrioritizedEpisodicReplayBuffer":
            rbuf = replay_buffers.PrioritizedEpisodicReplayBuffer(capacity=None)
        else:
            assert False

        # 2 transitions for env_id=0
        for _ in range(2):
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
        for _ in range(9):
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


class TestReplayBufferFail(unittest.TestCase):
    def setUp(self):
        self.rbuf = replay_buffers.PrioritizedReplayBuffer(100)
        self.trans1 = dict(
            state=0,
            action=1,
            reward=2,
            next_state=3,
            next_action=4,
            is_state_terminal=True,
        )
        self.rbuf.append(**self.trans1)

    def _sample1(self):
        self.rbuf.sample(1)

    def _set1(self):
        self.rbuf.update_errors([1.0])

    def test_fail_noupdate(self):
        self._sample1()
        self.assertRaises(AssertionError, self._sample1)

    def test_fail_update_first(self):
        self.assertRaises(AssertionError, self._set1)

    def test_fail_doubleupdate(self):
        self._sample1()
        self._set1()
        self.assertRaises(AssertionError, self._set1)


class TestBatchExperiences(unittest.TestCase):
    def test_batch_experiences(self):
        experiences = []
        experiences.append(
            [
                dict(
                    state=1,
                    action=1,
                    reward=1,
                    next_state=i,
                    next_action=1,
                    is_state_terminal=False,
                )
                for i in range(3)
            ]
        )
        experiences.append(
            [
                dict(
                    state=1,
                    action=1,
                    reward=1,
                    next_state=1,
                    next_action=1,
                    is_state_terminal=False,
                )
            ]
        )
        four_step_transition = [
            dict(
                state=1,
                action=1,
                reward=1,
                next_state=1,
                next_action=1,
                is_state_terminal=False,
            )
        ] * 3
        four_step_transition.append(
            dict(
                state=1,
                action=1,
                reward=1,
                next_state=5,
                next_action=1,
                is_state_terminal=True,
            )
        )
        experiences.append(four_step_transition)
        batch = replay_buffer.batch_experiences(
            experiences, torch.device("cpu"), lambda x: x, 0.99
        )
        self.assertEqual(batch["state"][0], 1)
        self.assertSequenceEqual(
            list(batch["is_state_terminal"]),
            list(np.asarray([0.0, 0.0, 1.0], dtype=np.float32)),
        )
        self.assertSequenceEqual(
            list(batch["discount"]),
            list(np.asarray([0.99 ** 3, 0.99 ** 1, 0.99 ** 4], dtype=np.float32)),
        )
        self.assertSequenceEqual(list(batch["next_state"]), list(np.asarray([2, 1, 5])))
