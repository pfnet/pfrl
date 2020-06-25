import torch

from pfrl.agents import pal
from pfrl.utils.recurrent import pack_and_forward


class DoublePAL(pal.PAL):
    def _compute_y_and_t(self, exp_batch):

        batch_state = exp_batch["state"]
        batch_size = len(exp_batch["reward"])

        if self.recurrent:
            qout, _ = pack_and_forward(
                self.model, batch_state, exp_batch["recurrent_state"],
            )
        else:
            qout = self.model(batch_state)

        batch_actions = exp_batch["action"]
        batch_q = qout.evaluate_actions(batch_actions)

        # Compute target values

        with torch.no_grad():
            batch_next_state = exp_batch["next_state"]
            if self.recurrent:
                next_qout, _ = pack_and_forward(
                    self.model, batch_next_state, exp_batch["next_recurrent_state"],
                )
                target_qout, _ = pack_and_forward(
                    self.target_model, batch_state, exp_batch["recurrent_state"],
                )
                target_next_qout, _ = pack_and_forward(
                    self.target_model,
                    batch_next_state,
                    exp_batch["next_recurrent_state"],
                )
            else:
                next_qout = self.model(batch_next_state)
                target_qout = self.target_model(batch_state)
                target_next_qout = self.target_model(batch_next_state)

            next_q_max = torch.reshape(
                target_next_qout.evaluate_actions(next_qout.greedy_actions),
                (batch_size,),
            )

            batch_rewards = exp_batch["reward"]
            batch_terminal = exp_batch["is_state_terminal"]

            # T Q: Bellman operator
            t_q = (
                batch_rewards
                + exp_batch["discount"] * (1.0 - batch_terminal) * next_q_max
            )

            # T_PAL Q: persistent advantage learning operator
            cur_advantage = torch.reshape(
                target_qout.compute_advantage(batch_actions), (batch_size,)
            )
            next_advantage = torch.reshape(
                target_next_qout.compute_advantage(batch_actions), (batch_size,)
            )
            tpal_q = t_q + self.alpha * torch.max(cur_advantage, next_advantage)

        return batch_q, tpal_q
