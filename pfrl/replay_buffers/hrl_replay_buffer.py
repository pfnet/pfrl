from pfrl.replay_buffers import ReplayBuffer


class LowerControllerReplayBuffer(ReplayBuffer):
    """Experience Replay Buffer for lower level controller in HRL.

    As described in
    https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf.

    Args:
        capacity (int): capacity in terms of number of transitions
        num_steps (int): Number of timesteps per stored transition
            (for N-step updates)
    """

    def __init__(self, capacity=None, num_steps=1):
        super().__init__(capacity, num_steps)

    def append(
        self,
        state,
        goal,
        action,
        reward,
        next_state=None,
        next_goal=None,
        next_action=None,
        is_state_terminal=False,
        env_id=0,
        **kwargs
    ):
        last_n_transitions = self.last_n_transitions[env_id]
        experience = dict(
            state=state,
            goal=goal,
            action=action,
            reward=reward,
            next_state=next_state,
            next_goal=next_goal,
            next_action=next_action,
            is_state_terminal=is_state_terminal,
            **kwargs
        )
        last_n_transitions.append(experience)
        if is_state_terminal:
            while last_n_transitions:
                self.memory.append(list(last_n_transitions))
                del last_n_transitions[0]
            assert len(last_n_transitions) == 0
        else:
            if len(last_n_transitions) == self.num_steps:
                self.memory.append(list(last_n_transitions))


class HigherControllerReplayBuffer(ReplayBuffer):
    """Experience Replay Buffer for higher level controller in HRL.

    As described in
    https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf.

    Args:
        capacity (int): capacity in terms of number of transitions
        num_steps (int): Number of timesteps per stored transition
            (for N-step updates)
    """

    def __init__(self, capacity=None, num_steps=1):
        super().__init__(capacity, num_steps)

    def append(
        self,
        state,
        goal,
        action,
        reward,
        next_state=None,
        next_action=None,
        is_state_terminal=False,
        state_arr=None,
        action_arr=None,
        env_id=0,
        **kwargs
    ):
        last_n_transitions = self.last_n_transitions[env_id]
        experience = dict(
            state=state,
            goal=goal,
            action=action,
            reward=reward,
            next_state=next_state,
            next_action=next_action,
            is_state_terminal=is_state_terminal,
            state_arr=state_arr,
            action_arr=action_arr,
            **kwargs
        )
        last_n_transitions.append(experience)
        if is_state_terminal:
            while last_n_transitions:
                self.memory.append(list(last_n_transitions))
                del last_n_transitions[0]
            assert len(last_n_transitions) == 0
        else:
            if len(last_n_transitions) == self.num_steps:
                self.memory.append(list(last_n_transitions))