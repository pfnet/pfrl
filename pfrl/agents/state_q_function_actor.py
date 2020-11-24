from logging import getLogger

import torch

from pfrl import agent
from pfrl.utils import evaluating
from pfrl.utils.batch_states import batch_states
from pfrl.utils.recurrent import (
    get_recurrent_state_at,
    one_step_forward,
    recurrent_state_as_numpy,
)


class StateQFunctionActor(agent.AsyncAgent):
    """Actor that acts according to the Q-function."""

    process_idx = None
    shared_attributes = ()

    def __init__(
        self,
        pipe,
        model,
        explorer,
        phi=lambda x: x,
        recurrent=False,
        logger=getLogger(__name__),
        batch_states=batch_states,
    ):
        self.pipe = pipe
        self.model = model
        self.explorer = explorer
        self.phi = phi
        self.recurrent = recurrent
        self.logger = logger
        self.batch_states = batch_states

        self.t = 0
        self.last_state = None
        self.last_action = None

        # Recurrent states of the model
        self.train_recurrent_states = None
        self.train_prev_recurrent_states = None
        self.test_recurrent_states = None

    @property
    def device(self):
        # Getting the device from the first layer of the model.
        # This is a work around since torch.nn.Module does not hold
        # a `device` attribute. For more details:
        # https://github.com/pytorch/pytorch/issues/7460
        return next(self.model.parameters()).device

    def _evaluate_model_and_update_recurrent_states(self, batch_obs):
        batch_xs = self.batch_states(batch_obs, self.device, self.phi)
        if self.recurrent:
            if self.training:
                self.train_prev_recurrent_states = self.train_recurrent_states
                batch_av, self.train_recurrent_states = one_step_forward(
                    self.model, batch_xs, self.train_recurrent_states
                )
            else:
                batch_av, self.test_recurrent_states = one_step_forward(
                    self.model, batch_xs, self.test_recurrent_states
                )
        else:
            batch_av = self.model(batch_xs)
        return batch_av

    def _send_to_learner(self, transition, stop_episode=False):
        self.pipe.send(("transition", transition))
        if stop_episode:
            self.pipe.send(("stop_episode", None))
            return self.pipe.recv()

    def act(self, obs):
        with torch.no_grad(), evaluating(self.model):
            action_value = self._evaluate_model_and_update_recurrent_states([obs])
            greedy_action = action_value.greedy_actions.detach().cpu().numpy()[0]
        if self.training:
            action = self.explorer.select_action(
                self.t, lambda: greedy_action, action_value=action_value
            )
            self.last_state = obs
            self.last_action = action
        else:
            action = greedy_action
        return action

    def observe(self, obs, reward, done, reset):
        if self.training:
            self.t += 1
            assert self.last_state is not None
            assert self.last_action is not None
            # Add a transition to the replay buffer
            transition = {
                "state": self.last_state,
                "action": self.last_action,
                "reward": reward,
                "next_state": obs,
                "is_state_terminal": done,
            }
            if self.recurrent:
                transition["recurrent_state"] = recurrent_state_as_numpy(
                    get_recurrent_state_at(
                        self.train_prev_recurrent_states, 0, detach=True
                    )
                )
                self.train_prev_recurrent_states = None
                transition["next_recurrent_state"] = recurrent_state_as_numpy(
                    get_recurrent_state_at(self.train_recurrent_states, 0, detach=True)
                )
            self._send_to_learner(transition, stop_episode=done or reset)
            if (done or reset) and self.recurrent:
                self.train_prev_recurrent_states = None
                self.train_recurrent_states = None
        else:
            if (done or reset) and self.recurrent:
                self.test_recurrent_states = None

    def save(self, dirname):
        self.pipe.send(("save", dirname))
        self.pipe.recv()

    def load(self, dirname):
        self.pipe.send(("load", dirname))
        self.pipe.recv()

    def get_statistics(self):
        self.pipe.send(("get_statistics", None))
        return self.pipe.recv()
