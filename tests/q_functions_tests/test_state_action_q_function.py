import unittest

import basetest_state_action_q_function as base
import pytest
import torch.nn.functional as F

import pfrl

assertions = unittest.TestCase("__init__")


@pytest.mark.parametrize("n_dim_obs", [1, 5])
@pytest.mark.parametrize("n_dim_action", [1, 3])
@pytest.mark.parametrize("n_hidden_layers", [0, 1, 2])
@pytest.mark.parametrize("n_hidden_channels", [1, 2])
@pytest.mark.parametrize("nonlinearity", ["relu", "elu"])
@pytest.mark.parametrize("last_wscale", [1, 1e-3])
class TestFCSAQFunction(base._TestSAQFunction):
    @pytest.fixture(autouse=True)
    def setUp(
        self,
        n_dim_obs,
        n_dim_action,
        n_hidden_layers,
        n_hidden_channels,
        nonlinearity,
        last_wscale,
    ):
        self.n_dim_obs = n_dim_obs
        self.n_dim_action = n_dim_action
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels
        self.nonlinearity = nonlinearity
        self.last_wscale = last_wscale

    def _test_call(self, gpu):
        nonlinearity = getattr(F, self.nonlinearity)
        model = pfrl.q_functions.FCSAQFunction(
            n_dim_obs=self.n_dim_obs,
            n_dim_action=self.n_dim_action,
            n_hidden_layers=self.n_hidden_layers,
            n_hidden_channels=self.n_hidden_channels,
            nonlinearity=nonlinearity,
            last_wscale=self.last_wscale,
        )
        self._test_call_given_model(model, gpu)

    def test_call_cpu(self):
        self._test_call(gpu=-1)

    @pytest.mark.gpu
    def test_call_gpu(self):
        self._test_call(gpu=0)


@pytest.mark.skip
@pytest.mark.parametrize("n_dim_obs", [1, 5])
@pytest.mark.parametrize("n_dim_action", [1, 3])
@pytest.mark.parametrize("n_hidden_layers", [0, 1, 2])
@pytest.mark.parametrize("n_hidden_channels", [1, 2])
@pytest.mark.parametrize("nonlinearity", ["relu", "elu"])
@pytest.mark.parametrize("last_wscale", [1, 1e-3])
class TestFCLSTMSAQFunction(base._TestSAQFunction):
    @pytest.fixture(autouse=True)
    def setUp(
        self,
        n_dim_obs,
        n_dim_action,
        n_hidden_layers,
        n_hidden_channels,
        nonlinearity,
        last_wscale,
    ):
        self.n_dim_obs = n_dim_obs
        self.n_dim_action = n_dim_action
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels
        self.nonlinearity = nonlinearity
        self.last_wscale = last_wscale

    def _test_call(self, gpu):
        nonlinearity = getattr(F, self.nonlinearity)
        model = pfrl.q_functions.FCLSTMSAQFunction(
            n_dim_obs=self.n_dim_obs,
            n_dim_action=self.n_dim_action,
            n_hidden_layers=self.n_hidden_layers,
            n_hidden_channels=self.n_hidden_channels,
            nonlinearity=nonlinearity,
            last_wscale=self.last_wscale,
        )
        self._test_call_given_model(model, gpu)

    def test_call_cpu(self):
        self._test_call(gpu=-1)

    @pytest.mark.gpu
    def test_call_gpu(self):
        self._test_call(gpu=0)


@pytest.mark.parametrize("n_dim_obs", [1, 5])
@pytest.mark.parametrize("n_dim_action", [1, 3])
@pytest.mark.parametrize("n_hidden_layers", [0, 1, 2])
@pytest.mark.parametrize("n_hidden_channels", [1, 2])
@pytest.mark.parametrize("normalize_input", [True, False])
@pytest.mark.parametrize("nonlinearity", ["relu", "elu"])
@pytest.mark.parametrize("last_wscale", [1, 1e-3])
class TestFCBNSAQFunction(base._TestSAQFunction):
    @pytest.fixture(autouse=True)
    def setUp(
        self,
        n_dim_obs,
        n_dim_action,
        n_hidden_layers,
        n_hidden_channels,
        normalize_input,
        nonlinearity,
        last_wscale,
    ):
        self.n_dim_obs = n_dim_obs
        self.n_dim_action = n_dim_action
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels
        self.normalize_input = normalize_input
        self.nonlinearity = nonlinearity
        self.last_wscale = last_wscale

    def _test_call(self, gpu):
        nonlinearity = getattr(F, self.nonlinearity)
        model = pfrl.q_functions.FCBNSAQFunction(
            n_dim_obs=self.n_dim_obs,
            n_dim_action=self.n_dim_action,
            n_hidden_layers=self.n_hidden_layers,
            n_hidden_channels=self.n_hidden_channels,
            normalize_input=self.normalize_input,
            nonlinearity=nonlinearity,
            last_wscale=self.last_wscale,
        )
        self._test_call_given_model(model, gpu)

    def test_call_cpu(self):
        self._test_call(gpu=-1)

    @pytest.mark.gpu
    def test_call_gpu(self):
        self._test_call(gpu=0)


@pytest.mark.parametrize("n_dim_obs", [1, 5])
@pytest.mark.parametrize("n_dim_action", [1, 3])
# LateAction requires n_hidden_layers >=1
@pytest.mark.parametrize("n_hidden_layers", [1, 2])
@pytest.mark.parametrize("n_hidden_channels", [1, 2])
@pytest.mark.parametrize("normalize_input", [True, False])
@pytest.mark.parametrize("nonlinearity", ["relu", "elu"])
@pytest.mark.parametrize("last_wscale", [1, 1e-3])
class TestFCBNLateActionSAQFunction(base._TestSAQFunction):
    @pytest.fixture(autouse=True)
    def setUp(
        self,
        n_dim_obs,
        n_dim_action,
        n_hidden_layers,
        n_hidden_channels,
        normalize_input,
        nonlinearity,
        last_wscale,
    ):
        self.n_dim_obs = n_dim_obs
        self.n_dim_action = n_dim_action
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels
        self.normalize_input = normalize_input
        self.nonlinearity = nonlinearity
        self.last_wscale = last_wscale

    def _test_call(self, gpu):
        nonlinearity = getattr(F, self.nonlinearity)
        model = pfrl.q_functions.FCBNLateActionSAQFunction(
            n_dim_obs=self.n_dim_obs,
            n_dim_action=self.n_dim_action,
            n_hidden_layers=self.n_hidden_layers,
            n_hidden_channels=self.n_hidden_channels,
            normalize_input=self.normalize_input,
            nonlinearity=nonlinearity,
            last_wscale=self.last_wscale,
        )
        self._test_call_given_model(model, gpu)

    def test_call_cpu(self):
        self._test_call(gpu=-1)

    @pytest.mark.gpu
    def test_call_gpu(self):
        self._test_call(gpu=0)


@pytest.mark.parametrize("n_dim_obs", [1, 5])
@pytest.mark.parametrize("n_dim_action", [1, 3])
# LateAction requires n_hidden_layers >=1
@pytest.mark.parametrize("n_hidden_layers", [1, 2])
@pytest.mark.parametrize("n_hidden_channels", [1, 2])
@pytest.mark.parametrize("nonlinearity", ["relu", "elu"])
@pytest.mark.parametrize("last_wscale", [1, 1e-3])
class TestFCLateActionSAQFunction(base._TestSAQFunction):
    @pytest.fixture(autouse=True)
    def setUp(
        self,
        n_dim_obs,
        n_dim_action,
        n_hidden_layers,
        n_hidden_channels,
        nonlinearity,
        last_wscale,
    ):
        self.n_dim_obs = n_dim_obs
        self.n_dim_action = n_dim_action
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels
        self.nonlinearity = nonlinearity
        self.last_wscale = last_wscale

    def _test_call(self, gpu):
        nonlinearity = getattr(F, self.nonlinearity)
        model = pfrl.q_functions.FCLateActionSAQFunction(
            n_dim_obs=self.n_dim_obs,
            n_dim_action=self.n_dim_action,
            n_hidden_layers=self.n_hidden_layers,
            n_hidden_channels=self.n_hidden_channels,
            nonlinearity=nonlinearity,
            last_wscale=self.last_wscale,
        )
        self._test_call_given_model(model, gpu)

    def test_call_cpu(self):
        self._test_call(gpu=-1)

    @pytest.mark.gpu
    def test_call_gpu(self):
        self._test_call(gpu=0)
