import unittest

import numpy as np
import torch

assertions = unittest.TestCase("__init__")


class _TestSAQFunction:
    def _test_call_given_model(self, model, gpu):
        # This method only check if a given model can receive random input
        # data and return output data with the correct interface.
        batch_size = 7
        obs = np.random.rand(batch_size, self.n_dim_obs)
        action = np.random.rand(batch_size, self.n_dim_action)
        obs, action = torch.from_numpy(obs).float(), torch.from_numpy(action).float()
        if gpu >= 0:
            model.to(torch.device("cuda", gpu))
            obs = obs.to(torch.device("cuda", gpu))
            action = action.to(torch.device("cuda", gpu))
        y = model(obs, action)
        assertions.assertTrue(isinstance(y, torch.Tensor))
        assertions.assertEqual(y.shape, (batch_size, 1))
        assertions.assertEqual(y.get_device(), obs.get_device())
