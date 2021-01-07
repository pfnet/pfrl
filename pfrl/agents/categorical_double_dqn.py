import pfrl
from pfrl.agents import categorical_dqn
from pfrl.agents.categorical_dqn import _apply_categorical_projection
from pfrl.utils.recurrent import pack_and_forward


class CategoricalDoubleDQN(categorical_dqn.CategoricalDQN):
    """Categorical Double DQN."""

    def _compute_target_values(self, exp_batch):
        """Compute a batch of target return distributions."""

        batch_next_state = exp_batch["next_state"]
        batch_rewards = exp_batch["reward"]
        batch_terminal = exp_batch["is_state_terminal"]

        with pfrl.utils.evaluating(self.target_model), pfrl.utils.evaluating(
            self.model
        ):
            if self.recurrent:
                target_next_qout, _ = pack_and_forward(
                    self.target_model,
                    batch_next_state,
                    exp_batch["next_recurrent_state"],
                )
                next_qout, _ = pack_and_forward(
                    self.model,
                    batch_next_state,
                    exp_batch["next_recurrent_state"],
                )
            else:
                target_next_qout = self.target_model(batch_next_state)
                next_qout = self.model(batch_next_state)

        batch_size = batch_rewards.shape[0]
        z_values = target_next_qout.z_values
        n_atoms = z_values.numel()

        # next_q_max: (batch_size, n_atoms)
        next_q_max = target_next_qout.evaluate_actions_as_distribution(
            next_qout.greedy_actions.detach()
        )
        assert next_q_max.shape == (batch_size, n_atoms), next_q_max.shape

        # Tz: (batch_size, n_atoms)
        Tz = (
            batch_rewards[..., None]
            + (1.0 - batch_terminal[..., None])
            * exp_batch["discount"][..., None]
            * z_values[None]
        )
        return _apply_categorical_projection(Tz, next_q_max, z_values)
