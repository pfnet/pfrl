import collections
import copy
import ctypes
import multiprocessing as mp
import multiprocessing.synchronize
import time
from logging import Logger, getLogger
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import pfrl
from pfrl import agent
from pfrl.action_value import ActionValue
from pfrl.explorer import Explorer
from pfrl.replay_buffer import (
    AbstractEpisodicReplayBuffer,
    ReplayUpdater,
    batch_experiences,
    batch_recurrent_experiences,
)
from pfrl.replay_buffers import PrioritizedReplayBuffer
from pfrl.utils.batch_states import batch_states
from pfrl.utils.contexts import evaluating
from pfrl.utils.copy_param import synchronize_parameters
from pfrl.utils.recurrent import (
    get_recurrent_state_at,
    mask_recurrent_state_at,
    one_step_forward,
    pack_and_forward,
    recurrent_state_as_numpy,
)


def _mean_or_nan(xs: Sequence[float]) -> float:
    """Return its mean a non-empty sequence, numpy.nan for a empty one."""
    return np.mean(xs) if xs else np.nan


def compute_value_loss(
    y: torch.Tensor,
    t: torch.Tensor,
    clip_delta: bool = True,
    batch_accumulator: str = "mean",
) -> torch.Tensor:
    """Compute a loss for value prediction problem.

    Args:
        y (torch.Tensor): Predicted values.
        t (torch.Tensor): Target values.
        clip_delta (bool): Use the Huber loss function with delta=1 if set True.
        batch_accumulator (str): 'mean' or 'sum'. 'mean' will use the mean of
            the loss values in a batch. 'sum' will use the sum.
    Returns:
        (torch.Tensor) scalar loss
    """
    assert batch_accumulator in ("mean", "sum")
    y = y.reshape(-1, 1)
    t = t.reshape(-1, 1)
    if clip_delta:
        return F.smooth_l1_loss(y, t, reduction=batch_accumulator)
    else:
        return F.mse_loss(y, t, reduction=batch_accumulator) / 2


def compute_weighted_value_loss(
    y: torch.Tensor,
    t: torch.Tensor,
    weights: torch.Tensor,
    clip_delta: bool = True,
    batch_accumulator: str = "mean",
) -> torch.Tensor:
    """Compute a loss for value prediction problem.

    Args:
        y (torch.Tensor): Predicted values.
        t (torch.Tensor): Target values.
        weights (torch.Tensor): Weights for y, t.
        clip_delta (bool): Use the Huber loss function with delta=1 if set True.
        batch_accumulator (str): 'mean' will divide loss by batchsize
    Returns:
        (torch.Tensor) scalar loss
    """
    assert batch_accumulator in ("mean", "sum")
    y = y.reshape(-1, 1)
    t = t.reshape(-1, 1)
    if clip_delta:
        losses = F.smooth_l1_loss(y, t, reduction="none")
    else:
        losses = F.mse_loss(y, t, reduction="none") / 2
    losses = losses.reshape(-1,)
    weights = weights.to(losses.device)
    loss_sum = torch.sum(losses * weights)
    if batch_accumulator == "mean":
        loss = loss_sum / y.shape[0]
    elif batch_accumulator == "sum":
        loss = loss_sum
    return loss


def _batch_reset_recurrent_states_when_episodes_end(
    batch_done: Sequence[bool], batch_reset: Sequence[bool], recurrent_states: Any
) -> Any:
    """Reset recurrent states when episodes end.

    Args:
        batch_done (array-like of bool): True iff episodes are terminal.
        batch_reset (array-like of bool): True iff episodes will be reset.
        recurrent_states (object): Recurrent state.

    Returns:
        object: New recurrent states.
    """
    indices_that_ended = [
        i
        for i, (done, reset) in enumerate(zip(batch_done, batch_reset))
        if done or reset
    ]
    if indices_that_ended:
        return mask_recurrent_state_at(recurrent_states, indices_that_ended)
    else:
        return recurrent_states


def make_target_model_as_copy(model: torch.nn.Module) -> torch.nn.Module:
    target_model = copy.deepcopy(model)

    def flatten_parameters(mod):
        if isinstance(mod, torch.nn.RNNBase):
            mod.flatten_parameters()

    # RNNBase.flatten_parameters must be called again after deep-copy.
    # See: https://discuss.pytorch.org/t/why-do-we-need-flatten-parameters-when-using-rnn-with-dataparallel/46506  # NOQA
    target_model.apply(flatten_parameters)
    # set target n/w to evaluate only.
    target_model.eval()
    return target_model


class DQN(agent.AttributeSavingMixin, agent.BatchAgent):
    """Deep Q-Network algorithm.

    Args:
        q_function (StateQFunction): Q-function
        optimizer (Optimizer): Optimizer that is already setup
        replay_buffer (ReplayBuffer): Replay buffer
        gamma (float): Discount factor
        explorer (Explorer): Explorer that specifies an exploration strategy.
        gpu (int): GPU device id if not None nor negative.
        replay_start_size (int): if the replay buffer's size is less than
            replay_start_size, skip update
        minibatch_size (int): Minibatch size
        update_interval (int): Model update interval in step
        target_update_interval (int): Target model update interval in step
        clip_delta (bool): Clip delta if set True
        phi (callable): Feature extractor applied to observations
        target_update_method (str): 'hard' or 'soft'.
        soft_update_tau (float): Tau of soft target update.
        n_times_update (int): Number of repetition of update
        batch_accumulator (str): 'mean' or 'sum'
        episodic_update_len (int or None): Subsequences of this length are used
            for update if set int and episodic_update=True
        logger (Logger): Logger used
        batch_states (callable): method which makes a batch of observations.
            default is `pfrl.utils.batch_states.batch_states`
        recurrent (bool): If set to True, `model` is assumed to implement
            `pfrl.nn.Recurrent` and is updated in a recurrent
            manner.
        max_grad_norm (float or None): Maximum L2 norm of the gradient used for
            gradient clipping. If set to None, the gradient is not clipped.
    """

    saved_attributes = ("model", "target_model", "optimizer")

    def __init__(
        self,
        q_function: torch.nn.Module,
        optimizer: torch.optim.Optimizer,  # type: ignore  # somehow mypy complains
        replay_buffer: pfrl.replay_buffer.AbstractReplayBuffer,
        gamma: float,
        explorer: Explorer,
        gpu: Optional[int] = None,
        replay_start_size: int = 50000,
        minibatch_size: int = 32,
        update_interval: int = 1,
        target_update_interval: int = 10000,
        clip_delta: bool = True,
        phi: Callable[[Any], Any] = lambda x: x,
        target_update_method: str = "hard",
        soft_update_tau: float = 1e-2,
        n_times_update: int = 1,
        batch_accumulator: str = "mean",
        episodic_update_len: Optional[int] = None,
        logger: Logger = getLogger(__name__),
        batch_states: Callable[
            [Sequence[Any], torch.device, Callable[[Any], Any]], Any
        ] = batch_states,
        recurrent: bool = False,
        max_grad_norm: Optional[float] = None,
    ):
        self.model = q_function

        if gpu is not None and gpu >= 0:
            assert torch.cuda.is_available()
            self.device = torch.device("cuda:{}".format(gpu))
            self.model.to(self.device)
        else:
            self.device = torch.device("cpu")

        self.replay_buffer = replay_buffer
        self.optimizer = optimizer
        self.gamma = gamma
        self.explorer = explorer
        self.gpu = gpu
        self.target_update_interval = target_update_interval
        self.clip_delta = clip_delta
        self.phi = phi
        self.target_update_method = target_update_method
        self.soft_update_tau = soft_update_tau
        self.batch_accumulator = batch_accumulator
        assert batch_accumulator in ("mean", "sum")
        self.logger = logger
        self.batch_states = batch_states
        self.recurrent = recurrent
        update_func: Callable[..., None]
        if self.recurrent:
            assert isinstance(self.replay_buffer, AbstractEpisodicReplayBuffer)
            update_func = self.update_from_episodes
        else:
            update_func = self.update
        self.replay_updater = ReplayUpdater(
            replay_buffer=replay_buffer,
            update_func=update_func,
            batchsize=minibatch_size,
            episodic_update=recurrent,
            episodic_update_len=episodic_update_len,
            n_times_update=n_times_update,
            replay_start_size=replay_start_size,
            update_interval=update_interval,
        )
        self.minibatch_size = minibatch_size
        self.episodic_update_len = episodic_update_len
        self.replay_start_size = replay_start_size
        self.update_interval = update_interval
        self.max_grad_norm = max_grad_norm

        assert (
            target_update_interval % update_interval == 0
        ), "target_update_interval should be a multiple of update_interval"

        self.t = 0
        self.optim_t = 0  # Compensate pytorch optim not having `t`
        self._cumulative_steps = 0
        self.target_model = make_target_model_as_copy(self.model)

        # Statistics
        self.q_record: collections.deque = collections.deque(maxlen=1000)
        self.loss_record: collections.deque = collections.deque(maxlen=100)

        # Recurrent states of the model
        self.train_recurrent_states: Any = None
        self.train_prev_recurrent_states: Any = None
        self.test_recurrent_states: Any = None

        # Error checking
        if (
            self.replay_buffer.capacity is not None
            and self.replay_buffer.capacity < self.replay_updater.replay_start_size
        ):
            raise ValueError("Replay start size cannot exceed replay buffer capacity.")

    @property
    def cumulative_steps(self) -> int:
        # cumulative_steps counts the overall steps during the training.
        return self._cumulative_steps

    def _setup_actor_learner_training(
        self, n_actors: int, actor_update_interval: int, update_counter: Any,
    ) -> Tuple[
        torch.nn.Module,
        Sequence[mp.connection.Connection],
        Sequence[mp.connection.Connection],
    ]:
        assert actor_update_interval > 0

        self.actor_update_interval = actor_update_interval
        self.update_counter = update_counter

        # Make a copy on shared memory and share among actors and the poller
        shared_model = copy.deepcopy(self.model).cpu()
        shared_model.share_memory()

        # Pipes are used for infrequent communication
        learner_pipes, actor_pipes = list(zip(*[mp.Pipe() for _ in range(n_actors)]))

        return (shared_model, learner_pipes, actor_pipes)

    def sync_target_network(self) -> None:
        """Synchronize target network with current network."""
        synchronize_parameters(
            src=self.model,
            dst=self.target_model,
            method=self.target_update_method,
            tau=self.soft_update_tau,
        )

    def update(
        self, experiences: List[List[Dict[str, Any]]], errors_out: Optional[list] = None
    ) -> None:
        """Update the model from experiences

        Args:
            experiences (list): List of lists of dicts.
                For DQN, each dict must contains:
                  - state (object): State
                  - action (object): Action
                  - reward (float): Reward
                  - is_state_terminal (bool): True iff next state is terminal
                  - next_state (object): Next state
                  - weight (float, optional): Weight coefficient. It can be
                    used for importance sampling.
            errors_out (list or None): If set to a list, then TD-errors
                computed from the given experiences are appended to the list.

        Returns:
            None
        """
        has_weight = "weight" in experiences[0][0]
        exp_batch = batch_experiences(
            experiences,
            device=self.device,
            phi=self.phi,
            gamma=self.gamma,
            batch_states=self.batch_states,
        )
        if has_weight:
            exp_batch["weights"] = torch.tensor(
                [elem[0]["weight"] for elem in experiences],
                device=self.device,
                dtype=torch.float32,
            )
            if errors_out is None:
                errors_out = []
        loss = self._compute_loss(exp_batch, errors_out=errors_out)
        if has_weight:
            assert isinstance(self.replay_buffer, PrioritizedReplayBuffer)
            self.replay_buffer.update_errors(errors_out)

        self.loss_record.append(float(loss.detach().cpu().numpy()))

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            pfrl.utils.clip_l2_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.optim_t += 1

    def update_from_episodes(
        self, episodes: List[List[Dict[str, Any]]], errors_out: Optional[list] = None
    ) -> None:
        assert errors_out is None, "Recurrent DQN does not support PrioritizedBuffer"
        episodes = sorted(episodes, key=len, reverse=True)
        exp_batch = batch_recurrent_experiences(
            episodes,
            device=self.device,
            phi=self.phi,
            gamma=self.gamma,
            batch_states=self.batch_states,
        )
        loss = self._compute_loss(exp_batch, errors_out=None)
        self.loss_record.append(float(loss.detach().cpu().numpy()))
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            pfrl.utils.clip_l2_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.optim_t += 1

    def _compute_target_values(self, exp_batch: Dict[str, Any]) -> torch.Tensor:
        batch_next_state = exp_batch["next_state"]

        if self.recurrent:
            target_next_qout, _ = pack_and_forward(
                self.target_model, batch_next_state, exp_batch["next_recurrent_state"],
            )
        else:
            target_next_qout = self.target_model(batch_next_state)
        next_q_max = target_next_qout.max

        batch_rewards = exp_batch["reward"]
        batch_terminal = exp_batch["is_state_terminal"]
        discount = exp_batch["discount"]

        return batch_rewards + discount * (1.0 - batch_terminal) * next_q_max

    def _compute_y_and_t(
        self, exp_batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = exp_batch["reward"].shape[0]

        # Compute Q-values for current states
        batch_state = exp_batch["state"]

        if self.recurrent:
            qout, _ = pack_and_forward(
                self.model, batch_state, exp_batch["recurrent_state"]
            )
        else:
            qout = self.model(batch_state)

        batch_actions = exp_batch["action"]
        batch_q = torch.reshape(qout.evaluate_actions(batch_actions), (batch_size, 1))

        with torch.no_grad():
            batch_q_target = torch.reshape(
                self._compute_target_values(exp_batch), (batch_size, 1)
            )

        return batch_q, batch_q_target

    def _compute_loss(
        self, exp_batch: Dict[str, Any], errors_out: Optional[list] = None
    ) -> torch.Tensor:
        """Compute the Q-learning loss for a batch of experiences


        Args:
          exp_batch (dict): A dict of batched arrays of transitions
        Returns:
          Computed loss from the minibatch of experiences
        """
        y, t = self._compute_y_and_t(exp_batch)

        self.q_record.extend(y.detach().cpu().numpy().ravel())

        if errors_out is not None:
            del errors_out[:]
            delta = torch.abs(y - t)
            if delta.ndim == 2:
                delta = torch.sum(delta, dim=1)
            delta = delta.detach().cpu().numpy()
            for e in delta:
                errors_out.append(e)

        if "weights" in exp_batch:
            return compute_weighted_value_loss(
                y,
                t,
                exp_batch["weights"],
                clip_delta=self.clip_delta,
                batch_accumulator=self.batch_accumulator,
            )
        else:
            return compute_value_loss(
                y,
                t,
                clip_delta=self.clip_delta,
                batch_accumulator=self.batch_accumulator,
            )

    def _evaluate_model_and_update_recurrent_states(
        self, batch_obs: Sequence[Any]
    ) -> ActionValue:
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

    def batch_act(self, batch_obs: Sequence[Any]) -> Sequence[Any]:
        with torch.no_grad(), evaluating(self.model):
            batch_av = self._evaluate_model_and_update_recurrent_states(batch_obs)
            batch_argmax = batch_av.greedy_actions.detach().cpu().numpy()
        if self.training:
            batch_action = [
                self.explorer.select_action(
                    self.t, lambda: batch_argmax[i], action_value=batch_av[i : i + 1],
                )
                for i in range(len(batch_obs))
            ]
            self.batch_last_obs = list(batch_obs)
            self.batch_last_action = list(batch_action)
        else:
            batch_action = batch_argmax
        return batch_action

    def _batch_observe_train(
        self,
        batch_obs: Sequence[Any],
        batch_reward: Sequence[float],
        batch_done: Sequence[bool],
        batch_reset: Sequence[bool],
    ) -> None:

        for i in range(len(batch_obs)):
            self.t += 1
            self._cumulative_steps += 1
            # Update the target network
            if self.t % self.target_update_interval == 0:
                self.sync_target_network()
            if self.batch_last_obs[i] is not None:
                assert self.batch_last_action[i] is not None
                # Add a transition to the replay buffer
                transition = {
                    "state": self.batch_last_obs[i],
                    "action": self.batch_last_action[i],
                    "reward": batch_reward[i],
                    "next_state": batch_obs[i],
                    "next_action": None,
                    "is_state_terminal": batch_done[i],
                }
                if self.recurrent:
                    transition["recurrent_state"] = recurrent_state_as_numpy(
                        get_recurrent_state_at(
                            self.train_prev_recurrent_states, i, detach=True
                        )
                    )
                    transition["next_recurrent_state"] = recurrent_state_as_numpy(
                        get_recurrent_state_at(
                            self.train_recurrent_states, i, detach=True
                        )
                    )
                self.replay_buffer.append(env_id=i, **transition)
                if batch_reset[i] or batch_done[i]:
                    self.batch_last_obs[i] = None
                    self.batch_last_action[i] = None
                    self.replay_buffer.stop_current_episode(env_id=i)
            self.replay_updater.update_if_necessary(self.t)

        if self.recurrent:
            # Reset recurrent states when episodes end
            self.train_prev_recurrent_states = None
            self.train_recurrent_states = _batch_reset_recurrent_states_when_episodes_end(  # NOQA
                batch_done=batch_done,
                batch_reset=batch_reset,
                recurrent_states=self.train_recurrent_states,
            )

    def _batch_observe_eval(
        self,
        batch_obs: Sequence[Any],
        batch_reward: Sequence[float],
        batch_done: Sequence[bool],
        batch_reset: Sequence[bool],
    ) -> None:
        if self.recurrent:
            # Reset recurrent states when episodes end
            self.test_recurrent_states = _batch_reset_recurrent_states_when_episodes_end(  # NOQA
                batch_done=batch_done,
                batch_reset=batch_reset,
                recurrent_states=self.test_recurrent_states,
            )

    def batch_observe(
        self,
        batch_obs: Sequence[Any],
        batch_reward: Sequence[float],
        batch_done: Sequence[bool],
        batch_reset: Sequence[bool],
    ) -> None:
        if self.training:
            return self._batch_observe_train(
                batch_obs, batch_reward, batch_done, batch_reset
            )
        else:
            return self._batch_observe_eval(
                batch_obs, batch_reward, batch_done, batch_reset
            )

    def _can_start_replay(self) -> bool:
        if len(self.replay_buffer) < self.replay_start_size:
            return False
        if self.recurrent:
            assert isinstance(self.replay_buffer, AbstractEpisodicReplayBuffer)
            if self.replay_buffer.n_episodes < self.minibatch_size:
                return False
        return True

    def _poll_pipe(
        self,
        actor_idx: int,
        pipe: mp.connection.Connection,
        replay_buffer_lock: mp.synchronize.Lock,
        exception_event: mp.synchronize.Event,
    ) -> None:
        if pipe.closed:
            return
        try:
            while pipe.poll() and not exception_event.is_set():
                cmd, data = pipe.recv()
                if cmd == "get_statistics":
                    assert data is None
                    with replay_buffer_lock:
                        stats = self.get_statistics()
                    pipe.send(stats)
                elif cmd == "load":
                    self.load(data)
                    pipe.send(None)
                elif cmd == "save":
                    self.save(data)
                    pipe.send(None)
                elif cmd == "transition":
                    with replay_buffer_lock:
                        if "env_id" not in data:
                            data["env_id"] = actor_idx
                        self.replay_buffer.append(**data)
                        self._cumulative_steps += 1
                elif cmd == "stop_episode":
                    idx = actor_idx if data is None else data
                    with replay_buffer_lock:
                        self.replay_buffer.stop_current_episode(env_id=idx)
                        stats = self.get_statistics()
                    pipe.send(stats)

                else:
                    raise RuntimeError("Unknown command from actor: {}".format(cmd))
        except EOFError:
            pipe.close()
        except Exception:
            self.logger.exception("Poller loop failed. Exiting")
            exception_event.set()

    def _learner_loop(
        self,
        shared_model: torch.nn.Module,
        pipes: Sequence[mp.connection.Connection],
        replay_buffer_lock: mp.synchronize.Lock,
        stop_event: mp.synchronize.Event,
        exception_event: mp.synchronize.Event,
        n_updates: Optional[int] = None,
        step_hooks: List[Callable[[None, agent.Agent, int], Any]] = [],
        optimizer_step_hooks: List[Callable[[None, agent.Agent, int], Any]] = [],
    ) -> None:
        try:
            update_counter = 0
            # To stop this loop, call stop_event.set()
            while not stop_event.is_set():
                # Update model if possible
                if not self._can_start_replay():
                    continue
                if n_updates is not None:
                    assert self.optim_t <= n_updates
                    if self.optim_t == n_updates:
                        stop_event.set()
                        break

                if self.recurrent:
                    assert isinstance(self.replay_buffer, AbstractEpisodicReplayBuffer)
                    with replay_buffer_lock:
                        episodes = self.replay_buffer.sample_episodes(
                            self.minibatch_size, self.episodic_update_len
                        )
                    self.update_from_episodes(episodes)
                else:
                    with replay_buffer_lock:
                        transitions = self.replay_buffer.sample(self.minibatch_size)

                    self.update(transitions)

                # Update the shared model. This can be expensive if GPU is used
                # since this is a DtoH copy, so it is updated only at regular
                # intervals.
                update_counter += 1
                if update_counter % self.actor_update_interval == 0:
                    with self.update_counter.get_lock():
                        self.update_counter.value += 1
                        shared_model.load_state_dict(self.model.state_dict())

                # To keep the ratio of target updates to model updates,
                # here we calculate back the effective current timestep
                # from update_interval and number of updates so far.
                effective_timestep = self.optim_t * self.update_interval
                # We can safely assign self.t since in the learner
                # it isn't updated by any other method
                self.t = effective_timestep

                for hook in optimizer_step_hooks:
                    hook(None, self, self.optim_t)

                for hook in step_hooks:
                    hook(None, self, effective_timestep)

                if effective_timestep % self.target_update_interval == 0:
                    self.sync_target_network()
        except Exception:
            self.logger.exception("Learner loop failed. Exiting")
            exception_event.set()

    def _poller_loop(
        self,
        shared_model: torch.nn.Module,
        pipes: Sequence[mp.connection.Connection],
        replay_buffer_lock: mp.synchronize.Lock,
        stop_event: mp.synchronize.Event,
        exception_event: mp.synchronize.Event,
    ) -> None:
        # To stop this loop, call stop_event.set()
        while not stop_event.is_set() and not exception_event.is_set():
            time.sleep(1e-6)
            # Poll actors for messages
            for i, pipe in enumerate(pipes):
                self._poll_pipe(i, pipe, replay_buffer_lock, exception_event)

    def setup_actor_learner_training(
        self,
        n_actors: int,
        update_counter: Optional[Any] = None,
        n_updates: Optional[int] = None,
        actor_update_interval: int = 8,
        step_hooks: List[Callable[[None, agent.Agent, int], Any]] = [],
        optimizer_step_hooks: List[Callable[[None, agent.Agent, int], Any]] = [],
    ):
        if update_counter is None:
            update_counter = mp.Value(ctypes.c_ulong)

        (shared_model, learner_pipes, actor_pipes) = self._setup_actor_learner_training(
            n_actors, actor_update_interval, update_counter
        )
        exception_event = mp.Event()

        def make_actor(i):
            return pfrl.agents.StateQFunctionActor(
                pipe=actor_pipes[i],
                model=shared_model,
                explorer=self.explorer,
                phi=self.phi,
                batch_states=self.batch_states,
                logger=self.logger,
                recurrent=self.recurrent,
            )

        replay_buffer_lock = mp.Lock()

        poller_stop_event = mp.Event()
        poller = pfrl.utils.StoppableThread(
            target=self._poller_loop,
            kwargs=dict(
                shared_model=shared_model,
                pipes=learner_pipes,
                replay_buffer_lock=replay_buffer_lock,
                stop_event=poller_stop_event,
                exception_event=exception_event,
            ),
            stop_event=poller_stop_event,
        )

        learner_stop_event = mp.Event()
        learner = pfrl.utils.StoppableThread(
            target=self._learner_loop,
            kwargs=dict(
                shared_model=shared_model,
                pipes=learner_pipes,
                replay_buffer_lock=replay_buffer_lock,
                stop_event=learner_stop_event,
                n_updates=n_updates,
                exception_event=exception_event,
                step_hooks=step_hooks,
                optimizer_step_hooks=optimizer_step_hooks,
            ),
            stop_event=learner_stop_event,
        )

        return make_actor, learner, poller, exception_event

    def stop_episode(self) -> None:
        if self.recurrent:
            self.test_recurrent_states = None

    def get_statistics(self):
        return [
            ("average_q", _mean_or_nan(self.q_record)),
            ("average_loss", _mean_or_nan(self.loss_record)),
            ("cumulative_steps", self.cumulative_steps),
            ("n_updates", self.optim_t),
            ("rlen", len(self.replay_buffer)),
        ]
