import gym
import numpy as np
import pytest

import pfrl


@pytest.mark.parametrize("num_envs", [1, 2, 3])
@pytest.mark.parametrize("env_id", ["CartPole-v0", "Pendulum-v0"])
@pytest.mark.parametrize("random_seed_offset", [0, 100])
@pytest.mark.parametrize(
    "vector_env_to_test", ["SerialVectorEnv", "MultiprocessVectorEnv"]
)
class TestSerialVectorEnv:
    @pytest.fixture(autouse=True)
    def setUp(self, num_envs, env_id, random_seed_offset, vector_env_to_test):
        self.num_envs = num_envs
        self.env_id = env_id
        self.random_seed_offset = random_seed_offset
        self.vector_env_to_test = vector_env_to_test
        # Init VectorEnv to test
        if self.vector_env_to_test == "SerialVectorEnv":
            self.vec_env = pfrl.envs.SerialVectorEnv(
                [gym.make(self.env_id) for _ in range(self.num_envs)]
            )
        elif self.vector_env_to_test == "MultiprocessVectorEnv":
            self.vec_env = pfrl.envs.MultiprocessVectorEnv(
                [(lambda: gym.make(self.env_id)) for _ in range(self.num_envs)]
            )
        else:
            assert False
        # Init envs to compare against
        self.envs = [gym.make(self.env_id) for _ in range(self.num_envs)]

    def teardown_method(self):
        # Delete so that all the subprocesses are joined
        del self.vec_env

    def test_num_envs(self):
        assert self.vec_env.num_envs == self.num_envs

    def test_action_space(self):
        assert self.vec_env.action_space == self.envs[0].action_space

    def test_observation_space(self):
        assert self.vec_env.observation_space == self.envs[0].observation_space

    def test_seed_reset_and_step(self):
        # seed
        seeds = [self.random_seed_offset + i for i in range(self.num_envs)]
        self.vec_env.seed(seeds)
        for env, seed in zip(self.envs, seeds):
            env.seed(seed)

        # reset
        obss = self.vec_env.reset()
        real_obss = [env.reset() for env in self.envs]
        np.testing.assert_allclose(obss, real_obss)

        # step
        actions = [env.action_space.sample() for env in self.envs]
        real_obss, real_rewards, real_dones, real_infos = zip(
            *[env.step(action) for env, action in zip(self.envs, actions)]
        )
        obss, rewards, dones, infos = self.vec_env.step(actions)
        np.testing.assert_allclose(obss, real_obss)
        assert rewards == real_rewards
        assert dones == real_dones
        assert infos == real_infos

        # reset with full mask should have no effect
        mask = np.ones(self.num_envs)
        obss = self.vec_env.reset(mask)
        np.testing.assert_allclose(obss, real_obss)

        # reset with partial mask
        mask = np.zeros(self.num_envs)
        mask[-1] = 1
        obss = self.vec_env.reset(mask)
        real_obss = list(real_obss)
        for i in range(self.num_envs):
            if not mask[i]:
                real_obss[i] = self.envs[i].reset()
        np.testing.assert_allclose(obss, real_obss)
