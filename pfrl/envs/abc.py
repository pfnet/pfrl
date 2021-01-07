import numpy as np
from gym import spaces

from pfrl import env


class ABC(env.Env):
    """Toy problem only for testing RL agents.

    # State space

    The state space consists of N discrete states plus the terminal state.

    # Observation space

    Observations are one-hot vectors that represents which state it is now.

        state 0 -> observation [1, 0, ..., 0]
        state 1 -> observation [0, 1, ..., 0]
        ...

    The size of an observation vector is N+2: N non-terminal states, the
    single terminal state and an additional dimension which is used in
    partially observable settings.

    # Action space

    In discrete-action settings, the action space consists of N discrete
    actions.

    In continuous-action settings, the action space is represented by N real
    numbers in [-1, 1]. Each action vector is interpreted as logits that
    defines probabilities of N discrete inner actions via the softmax function.

    # Dynamics

    Each episode starts from state 0.

    On state n, only the correct action (action n) will make a transition to
    state n+1.

    When the full set of correct actions are taken,
        - it moves to the terminal state with reward +1 (episodic), or
        - it moves back to the initial state with reward +1 (non-episodic).

    When it receives a wrong action,
        - it moves to the terminal state (episodic), or
        - it stays at the last state (non-episodic).

    The optimal policy is:

        state 0 -> action 0
        state 1 -> action 1
        ...

    Args:
        size (int): Size of the problem. It is equal to how many correct
            actions is required to get reward +1.
        discrete (bool): If set to True, use the discrete action space.
            If set to False, use the continuous action space that represents
            logits of probabilities of discrete actions.
        partially_observable (bool): If set to True, for some random episodes
            observation vectors are shifted so that state 0 is observed as
            [0, 1, ..., 0], which requires the agent to learn to see if
            observations from the current episode is shifted or not by
            remembering the initial observation.
        episodic (bool): If set to True, use episodic settings. If set to
            False, use non-episodic settings. See the explanation above.
        deterministic (bool): If set to True, everything will be deterministic.
            In continuous-action settings, most probable actions are taken
            instead of random samples for continuous-action. In partially
            observable settings, the n-th episode uses non-shifted obsevations
            if n is odd, otherwise uses shifted observations.
    """

    def __init__(
        self,
        size=2,
        discrete=True,
        partially_observable=False,
        episodic=True,
        deterministic=False,
    ):
        self.size = size
        self.terminal_state = size
        self.episodic = episodic
        self.partially_observable = partially_observable
        self.deterministic = deterministic
        self.n_max_offset = 1
        # (s_0, ..., s_N) + terminal state + offset
        self.n_dim_obs = self.size + 1 + self.n_max_offset
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_dim_obs,),
            dtype=np.float32,
        )
        if discrete:
            self.action_space = spaces.Discrete(self.size)
        else:
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.size,),
                dtype=np.float32,
            )

    def observe(self):
        state_vec = np.zeros((self.n_dim_obs,), dtype=np.float32)
        state_vec[self._state + self._offset] = 1.0
        return state_vec

    def reset(self):
        self._state = 0
        if self.partially_observable:
            # For partially observable settings, observations are shifted by
            # episode-dependent some offsets.
            if self.deterministic:
                self._offset = (getattr(self, "_offset", 0) + 1) % (
                    self.n_max_offset + 1
                )
            else:
                self._offset = np.random.randint(self.n_max_offset + 1)
        else:
            self._offset = 0
        return self.observe()

    def step(self, action):
        if isinstance(self.action_space, spaces.Box):
            assert isinstance(action, np.ndarray)
            action = np.clip(action, self.action_space.low, self.action_space.high)
            if self.deterministic:
                action = np.argmax(action)
            else:
                prob = np.exp(action) / np.exp(action).sum()
                action = np.random.choice(range(self.size), p=prob)
        reward = 0
        done = False
        if action == self._state:
            # Correct
            if self._state == self.size - 1:
                # Goal
                reward = 1.0
                if self.episodic:
                    # Terminal
                    done = True
                    self._state = self.terminal_state
                else:
                    # Restart
                    self._state = 0
            else:
                self._state += 1
        else:
            # Incorrect
            if self.episodic:
                # Terminal
                done = True
                self._state = self.terminal_state
        return self.observe(), reward, done, {}

    def close(self):
        pass
