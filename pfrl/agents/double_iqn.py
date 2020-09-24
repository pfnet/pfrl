import torch

from pfrl.agents import iqn
from pfrl.utils import evaluating
from pfrl.utils.recurrent import pack_and_forward


class DoubleIQN(iqn.IQN):

    """IQN with DoubleDQN-like target computation.

    For computing targets, rather than have the target network
    output the Q-value of its highest-valued action, the
    target network outputs the Q-value of the primary network’s
    highest valued action.
    """

    def _compute_target_values(self, exp_batch):
        """Compute a batch of target return distributions.

        Returns:
            torch.Tensor: (batch_size, N_prime).
        """
        batch_next_state = exp_batch["next_state"]
        batch_size = len(exp_batch["reward"])

        taus_tilde = torch.rand(
            batch_size,
            self.quantile_thresholds_K,
            device=self.device,
            dtype=torch.float,
        )
        with evaluating(self.model):
            if self.recurrent:
                next_tau2av, _ = pack_and_forward(
                    self.model, batch_next_state, exp_batch["next_recurrent_state"],
                )
            else:
                next_tau2av = self.model(batch_next_state)
        greedy_actions = next_tau2av(taus_tilde).greedy_actions

        taus_prime = torch.rand(
            batch_size,
            self.quantile_thresholds_N_prime,
            device=self.device,
            dtype=torch.float,
        )
        if self.recurrent:
            target_next_tau2av, _ = pack_and_forward(
                self.target_model, batch_next_state, exp_batch["next_recurrent_state"],
            )
        else:
            target_next_tau2av = self.target_model(batch_next_state)
        target_next_maxz = target_next_tau2av(taus_prime).evaluate_actions_as_quantiles(
            greedy_actions
        )

        batch_rewards = exp_batch["reward"]
        batch_terminal = exp_batch["is_state_terminal"]
        batch_discount = exp_batch["discount"]
        assert batch_rewards.shape == (batch_size,)
        assert batch_terminal.shape == (batch_size,)
        assert batch_discount.shape == (batch_size,)
        batch_rewards = batch_rewards.unsqueeze(-1)
        batch_terminal = batch_terminal.unsqueeze(-1)
        batch_discount = batch_discount.unsqueeze(-1)

        return (
            batch_rewards + batch_discount * (1.0 - batch_terminal) * target_next_maxz
        )
