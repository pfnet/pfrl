import torch

from pfrl.agents import dqn
from pfrl.utils.recurrent import pack_and_forward


def _apply_categorical_projection(y, y_probs, z):
    """Apply categorical projection.

    See Algorithm 1 in https://arxiv.org/abs/1707.06887.

    Args:
        y (ndarray): Values of atoms before projection. Its shape must be
            (batch_size, n_atoms).
        y_probs (ndarray): Probabilities of atoms whose values are y.
            Its shape must be (batch_size, n_atoms).
        z (ndarray): Values of atoms after projection. Its shape must be
            (n_atoms,). It is assumed that the values are sorted in ascending
            order and evenly spaced.

    Returns:
        ndarray: Probabilities of atoms whose values are z.
    """
    batch_size, n_atoms = y.shape
    assert z.shape == (n_atoms,)
    assert y_probs.shape == (batch_size, n_atoms)
    delta_z = z[1] - z[0]
    v_min = z[0]
    v_max = z[-1]
    y = torch.clamp(y, v_min, v_max)

    # bj: (batch_size, n_atoms)
    bj = (y - v_min) / delta_z
    assert bj.shape == (batch_size, n_atoms)
    # Avoid the error caused by inexact delta_z
    bj = torch.clamp(bj, 0, n_atoms - 1)

    # l, u: (batch_size, n_atoms)
    l, u = torch.floor(bj), torch.ceil(bj)
    assert l.shape == (batch_size, n_atoms)
    assert u.shape == (batch_size, n_atoms)

    z_probs = torch.zeros((batch_size, n_atoms), dtype=torch.float32, device=y.device)
    offset = torch.arange(
        0, batch_size * n_atoms, n_atoms, dtype=torch.int32, device=y.device
    )[..., None]
    # Accumulate m_l
    # Note that u - bj in the original paper is replaced with 1 - (bj - l) to
    # deal with the case when bj is an integer, i.e., l = u = bj
    z_probs.view(-1).scatter_add_(
        0, (l.long() + offset).view(-1), (y_probs * (1 - (bj - l))).view(-1)
    )
    # Accumulate m_u
    z_probs.view(-1).scatter_add_(
        0, (u.long() + offset).view(-1), (y_probs * (bj - l)).view(-1)
    )
    return z_probs


def compute_value_loss(eltwise_loss, batch_accumulator="mean"):
    """Compute a loss for value prediction problem.

    Args:
        eltwise_loss (Variable): Element-wise loss per example per atom
        batch_accumulator (str): 'mean' or 'sum'. 'mean' will use the mean of
            the loss values in a batch. 'sum' will use the sum.
    Returns:
        (Variable) scalar loss
    """
    assert batch_accumulator in ("mean", "sum")

    if batch_accumulator == "sum":
        loss = eltwise_loss.sum()
    else:
        loss = eltwise_loss.sum(dim=1).mean()
    return loss


def compute_weighted_value_loss(
    eltwise_loss, batch_size, weights, batch_accumulator="mean"
):
    """Compute a loss for value prediction problem.

    Args:
        eltwise_loss (Variable): Element-wise loss per example per atom
        weights (ndarray): Weights for y, t.
        batch_accumulator (str): 'mean' will divide loss by batchsize
    Returns:
        (Variable) scalar loss
    """
    assert batch_accumulator in ("mean", "sum")

    # eltwise_loss is (batchsize, n_atoms) array of losses
    # weights is an array of shape (batch_size)
    # sum loss across atoms and then apply weight per example in batch
    weights = weights.to(eltwise_loss.device)
    loss_sum = torch.matmul(eltwise_loss.sum(dim=1), weights)
    if batch_accumulator == "mean":
        loss = loss_sum / batch_size
    elif batch_accumulator == "sum":
        loss = loss_sum
    return loss


class CategoricalDQN(dqn.DQN):
    """Categorical DQN.

    See https://arxiv.org/abs/1707.06887.

    Arguments are the same as those of DQN except q_function must return
    DistributionalDiscreteActionValue and clip_delta is ignored.
    """

    def _compute_target_values(self, exp_batch):
        """Compute a batch of target return distributions."""

        batch_next_state = exp_batch["next_state"]
        if self.recurrent:
            target_next_qout, _ = pack_and_forward(
                self.target_model,
                batch_next_state,
                exp_batch["next_recurrent_state"],
            )
        else:
            target_next_qout = self.target_model(batch_next_state)

        batch_rewards = exp_batch["reward"]
        batch_terminal = exp_batch["is_state_terminal"]

        batch_size = exp_batch["reward"].shape[0]
        z_values = target_next_qout.z_values
        n_atoms = z_values.size()[0]

        # next_q_max: (batch_size, n_atoms)
        next_q_max = target_next_qout.max_as_distribution.detach()
        assert next_q_max.shape == (batch_size, n_atoms), next_q_max.shape

        # Tz: (batch_size, n_atoms)
        Tz = (
            batch_rewards[..., None]
            + (1.0 - batch_terminal[..., None])
            * torch.unsqueeze(exp_batch["discount"], 1)
            * z_values[None]
        )
        return _apply_categorical_projection(Tz, next_q_max, z_values)

    def _compute_y_and_t(self, exp_batch):
        """Compute a batch of predicted/target return distributions."""

        batch_size = exp_batch["reward"].shape[0]

        # Compute Q-values for current states
        batch_state = exp_batch["state"]

        # (batch_size, n_actions, n_atoms)
        if self.recurrent:
            qout, _ = pack_and_forward(
                self.model, batch_state, exp_batch["recurrent_state"]
            )
        else:
            qout = self.model(batch_state)
        n_atoms = qout.z_values.size()[0]

        batch_actions = exp_batch["action"]
        batch_q = qout.evaluate_actions_as_distribution(batch_actions)
        assert batch_q.shape == (batch_size, n_atoms)

        with torch.no_grad():
            batch_q_target = self._compute_target_values(exp_batch)
            assert batch_q_target.shape == (batch_size, n_atoms)

            # for `agent.get_statistics()`
            batch_q_scalars = qout.evaluate_actions(batch_actions)
            self.q_record.extend(batch_q_scalars.detach().cpu().numpy().ravel())

        return batch_q, batch_q_target

    def _compute_loss(self, exp_batch, errors_out=None):
        """Compute a loss of categorical DQN."""
        y, t = self._compute_y_and_t(exp_batch)
        # Minimize the cross entropy
        # y is clipped to avoid log(0)
        eltwise_loss = -t * torch.log(torch.clamp(y, 1e-10, 1.0))

        if errors_out is not None:
            del errors_out[:]
            # The loss per example is the sum of the atom-wise loss
            # Prioritization by KL-divergence
            delta = eltwise_loss.sum(dim=1)
            delta = delta.detach().cpu().numpy()
            for e in delta:
                errors_out.append(e)

        if "weights" in exp_batch:
            return compute_weighted_value_loss(
                eltwise_loss,
                y.shape[0],
                exp_batch["weights"],
                batch_accumulator=self.batch_accumulator,
            )
        else:
            return compute_value_loss(
                eltwise_loss, batch_accumulator=self.batch_accumulator
            )
