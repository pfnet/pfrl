import torch
from torch import nn
from torch import distributions
import numpy as np

import pfrl

from pfrl import explorers
from pfrl.replay_buffer import high_level_batch_experiences_with_goal
from pfrl.agents import HIROHighLevelGoalConditionedTD3, GoalConditionedTD3
from pfrl.nn import ConstantsMult
from pfrl.nn.lmbda import Lambda


class HRLControllerBase():
    def __init__(
            self,
            state_dim,
            goal_dim,
            action_dim,
            scale,
            replay_buffer,
            actor_lr,
            critic_lr,
            expl_noise,
            policy_noise,
            noise_clip,
            gamma,
            policy_freq,
            tau,
            is_low_level,
            buffer_freq,
            minibatch_size,
            gpu,
            add_entropy,
            burnin_action_func=None,
            replay_start_size=2500):
        self.scale = scale
        self.device = torch.device(f'cuda:{gpu}')
        self.scale_tensor = torch.tensor(self.scale).float().to(self.device)
        # parameters
        self.expl_noise = expl_noise
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.gamma = gamma
        self.policy_freq = policy_freq
        self.tau = tau
        self.is_low_level = is_low_level
        self.minibatch_size = minibatch_size
        self.add_entropy = add_entropy

        # create agent
        if self.add_entropy:
            def squashed_diagonal_gaussian_head(x):
                """
                taken from the SAC code.
                """
                assert x.shape[-1] == action_dim * 2
                mean, log_scale = torch.chunk(x, 2, dim=1)
                log_scale = torch.clamp(log_scale, -20.0, 2.0)
                var = torch.exp(log_scale * 2)
                base_distribution = distributions.Independent(
                    distributions.Normal(loc=mean, scale=torch.sqrt(var)), 1
                )
                # cache_size=1 is required for numerical stability
                return distributions.transformed_distribution.TransformedDistribution(
                    base_distribution, [distributions.transforms.TanhTransform(cache_size=1)]
                )

            # SAC policy definition:
            policy = nn.Sequential(
                nn.Linear(state_dim + goal_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, action_dim * 2),
                Lambda(squashed_diagonal_gaussian_head),
                )

            torch.nn.init.xavier_uniform_(policy[0].weight)
            torch.nn.init.xavier_uniform_(policy[2].weight)
            torch.nn.init.xavier_uniform_(policy[4].weight)

        else:
            policy = nn.Sequential(
                nn.Linear(state_dim + goal_dim, 300),
                nn.ReLU(),
                nn.Linear(300, 300),
                nn.ReLU(),
                nn.Linear(300, action_dim),
                nn.Tanh(),
                ConstantsMult(self.scale_tensor),
                pfrl.policies.DeterministicHead(),
                )

        policy_optimizer = torch.optim.Adam(policy.parameters(), lr=actor_lr)

        def make_q_func_with_optimizer():
            q_func = nn.Sequential(
                pfrl.nn.ConcatObsAndAction(),
                nn.Linear(state_dim + goal_dim + action_dim, 300),
                nn.ReLU(),
                nn.Linear(300, 300),
                nn.ReLU(),
                nn.Linear(300, 1),
            )
            q_func_optimizer = torch.optim.Adam(q_func.parameters(), lr=critic_lr)
            return q_func, q_func_optimizer

        q_func1, q_func1_optimizer = make_q_func_with_optimizer()
        q_func2, q_func2_optimizer = make_q_func_with_optimizer()

        # TODO - have proper low and high values from action space.
        # from the hiro paper, the scale is 1.0
        explorer = explorers.AdditiveGaussian(
            scale=self.expl_noise,
            low=-self.scale,
            high=self.scale
        )

        def default_target_policy_smoothing_func(batch_action):
            """Add noises to actions for target policy smoothing."""
            noise = torch.clamp(self.policy_noise * torch.randn_like(batch_action), -self.noise_clip, self.noise_clip)
            smoothed_action = batch_action + noise
            smoothed_action = torch.min(smoothed_action, torch.tensor(self.scale).to(self.device).float())
            smoothed_action = torch.max(smoothed_action, torch.tensor(-self.scale).to(self.device).float())
            return smoothed_action

        input_scale = self.scale_tensor if self.add_entropy else 1

        if self.is_low_level:
            # standard goal conditioned td3
            self.agent = GoalConditionedTD3(
                policy,
                q_func1,
                q_func2,
                policy_optimizer,
                q_func1_optimizer,
                q_func2_optimizer,
                replay_buffer,
                gamma=gamma,
                soft_update_tau=tau,
                explorer=explorer,
                update_interval=1,
                policy_update_delay=policy_freq,
                replay_start_size=replay_start_size,
                buffer_freq=buffer_freq,
                minibatch_size=minibatch_size,
                gpu=gpu,
                add_entropy=self.add_entropy,
                scale=input_scale,
                burnin_action_func=burnin_action_func,
                target_policy_smoothing_func=default_target_policy_smoothing_func
                )
        else:
            self.agent = HIROHighLevelGoalConditionedTD3(
                policy,
                q_func1,
                q_func2,
                policy_optimizer,
                q_func1_optimizer,
                q_func2_optimizer,
                replay_buffer,
                gamma=gamma,
                soft_update_tau=tau,
                explorer=explorer,
                update_interval=1,
                policy_update_delay=policy_freq,
                replay_start_size=replay_start_size/buffer_freq - 5,
                buffer_freq=buffer_freq,
                minibatch_size=minibatch_size,
                gpu=gpu,
                add_entropy=self.add_entropy,
                scale=input_scale,
                burnin_action_func=burnin_action_func,
                target_policy_smoothing_func=default_target_policy_smoothing_func
                )

        self.device = self.agent.device

    def save(self, directory):
        """
        save the internal state of the TD3 agent.
        """
        self.agent.save(directory)

    def load(self, directory):
        """
        load the internal state of the TD3 agent.
        """
        self.agent.load(directory)

    def policy(self, state, goal):
        """
        run the policy (actor).
        """
        action = self.agent.act_with_goal(torch.FloatTensor(state), torch.FloatTensor(goal))
        return np.clip(action, a_min=-self.scale, a_max=self.scale)

    def _observe(self, states, goals, rewards, done, state_arr=None, action_arr=None):
        """
        observe, and train (if we can sample from the replay buffer)
        """
        self.agent.observe_with_goal(torch.FloatTensor(states), torch.FloatTensor(goals), rewards, done, None)

    def observe(self, states, goals, rewards, done, iterations=1):
        """
        get data from the replay buffer, and train.
        """
        return self._observe(states, goals, rewards, goals, done)


# lower controller
class LowerController(HRLControllerBase):
    def __init__(
            self,
            state_dim,
            goal_dim,
            action_dim,
            scale,
            replay_buffer,
            add_entropy,
            actor_lr=0.0001,
            critic_lr=0.001,
            expl_noise=1.0,
            policy_noise=0.2,
            noise_clip=0.5,
            gamma=0.99,
            policy_freq=2,
            tau=0.005,
            is_low_level=True,
            buffer_freq=10,
            minibatch_size=100,
            gpu=None,
            burnin_action_func=None):
        super(LowerController, self).__init__(
                                            state_dim=state_dim,
                                            goal_dim=goal_dim,
                                            action_dim=action_dim,
                                            scale=scale,
                                            replay_buffer=replay_buffer,
                                            actor_lr=actor_lr,
                                            critic_lr=critic_lr,
                                            expl_noise=expl_noise,
                                            policy_noise=policy_noise,
                                            noise_clip=noise_clip,
                                            gamma=gamma,
                                            policy_freq=policy_freq,
                                            tau=tau,
                                            is_low_level=is_low_level,
                                            buffer_freq=buffer_freq,
                                            minibatch_size=minibatch_size,
                                            gpu=gpu,
                                            add_entropy=add_entropy,
                                            burnin_action_func=burnin_action_func)

    def observe(self, n_s, g, r, done):

        return self._observe(n_s, g, r, done)


# higher controller

class HigherController(HRLControllerBase):
    def __init__(
            self,
            state_dim,
            goal_dim,
            action_dim,
            scale,
            replay_buffer,
            add_entropy,
            actor_lr=0.0001,
            critic_lr=0.001,
            expl_noise=1.0,
            policy_noise=0.2,
            noise_clip=0.5,
            gamma=0.99,
            policy_freq=2,
            tau=0.005,
            is_low_level=False,
            buffer_freq=10,
            minibatch_size=100,
            gpu=None,
            burnin_action_func=None):
        super(HigherController, self).__init__(
                                                state_dim=state_dim,
                                                goal_dim=goal_dim,
                                                action_dim=action_dim,
                                                scale=scale,
                                                replay_buffer=replay_buffer,
                                                actor_lr=actor_lr,
                                                critic_lr=critic_lr,
                                                expl_noise=expl_noise,
                                                policy_noise=policy_noise,
                                                noise_clip=noise_clip,
                                                gamma=gamma,
                                                policy_freq=policy_freq,
                                                tau=tau,
                                                is_low_level=is_low_level,
                                                buffer_freq=buffer_freq,
                                                minibatch_size=minibatch_size,
                                                gpu=gpu,
                                                add_entropy=add_entropy,
                                                burnin_action_func=burnin_action_func)
        self.action_dim = action_dim

    def _off_policy_corrections(self, low_con, batch_size, sgoals, states, actions, candidate_goals=8):
        """
        implementation of the novel off policy correction in the HIRO paper.
        """

        first_s = [s[0] for s in states]  # First x
        last_s = [s[-1] for s in states]  # Last x

        # Shape: (batch_size, 1, subgoal_dim)
        # diff = 1
        # different in goals
        diff_goal = (np.array(last_s) -
                     np.array(first_s))[:, np.newaxis, :self.action_dim]

        # Shape: (batch_size, 1, subgoal_dim)
        # original = 1
        # random = candidate_goals
        original_goal = np.array(sgoals)[:, np.newaxis, :]
        # select random goals
        random_goals = np.random.normal(loc=diff_goal, scale=.5*self.scale[None, None, :],
                                        size=(batch_size, candidate_goals, original_goal.shape[-1]))
        random_goals = random_goals.clip(-self.scale, self.scale)

        # Shape: (batch_size, 10, subgoal_dim)
        candidates = np.concatenate([original_goal, diff_goal, random_goals], axis=1)
        # states = np.array(states)[:, :-1, :]
        actions = np.array(actions)
        seq_len = len(states[0])

        # For ease
        new_batch_sz = seq_len * batch_size
        action_dim = actions[0][0].shape
        obs_dim = states[0][0].shape
        ncands = candidates.shape[1]

        true_actions = actions.reshape((new_batch_sz,) + action_dim)
        observations = states.reshape((new_batch_sz,) + obs_dim)
        goal_shape = (new_batch_sz, self.action_dim)
        # observations = get_obs_tensor(observations, sg_corrections=True)

        # batched_candidates = np.tile(candidates, [seq_len, 1, 1])
        # batched_candidates = batched_candidates.transpose(1, 0, 2)

        policy_actions = np.zeros((ncands, new_batch_sz) + action_dim)
        low_con.agent.training = False
        for c in range(ncands):
            subgoal = candidates[:,c]
            candidate = (subgoal + states[:, 0, :self.action_dim])[:, None] - states[:, :, :self.action_dim]
            candidate = candidate.reshape(*goal_shape)
            policy_actions[c] = low_con.policy(torch.tensor(observations).float(), torch.tensor(candidate).float())
        low_con.agent.training = True

        difference = (policy_actions - true_actions)
        difference = np.where(difference != -np.inf, difference, 0)
        difference = difference.reshape((ncands, batch_size, seq_len) + action_dim).transpose(1, 0, 2, 3)

        logprob = -0.5*np.sum(np.linalg.norm(difference, axis=-1)**2, axis=-1)
        max_indices = np.argmax(logprob, axis=-1)
        # return best candidates with maximum probability
        return candidates[np.arange(batch_size), max_indices]

    def update(self, low_con):
        batch = self.agent.sample_if_possible()
        if batch:
            experience = high_level_batch_experiences_with_goal(batch, self.device, lambda x: x, self.gamma)
            actions = experience['action']
            action_arr = experience['action_arr']
            state_arr = experience['state_arr']
            actions = self._off_policy_corrections(
                low_con,
                self.minibatch_size,
                actions.cpu().data.numpy(),
                state_arr.cpu().data.numpy(),
                action_arr.cpu().data.numpy())

            tensor_actions = torch.FloatTensor(actions).to(self.agent.device)
            # relabel actions
            experience['action'] = tensor_actions

            self.agent.high_level_update_batch(experience)

    def observe(self, state_arr, action_arr, r, g, n_s, done):
        """
        train the high level controller with
        the novel off policy correction.
        """
        # step 1 - record experience in replay buffer
        self.agent.observe_with_goal_state_action_arr(torch.FloatTensor(state_arr),
                                                      torch.FloatTensor(action_arr),
                                                      torch.FloatTensor(n_s),
                                                      torch.FloatTensor(g), r, done, None)
