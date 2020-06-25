from pfrl import explorer


class Greedy(explorer.Explorer):
    """No exploration"""

    def select_action(self, t, greedy_action_func, action_value=None):
        return greedy_action_func()

    def __repr__(self):
        return "Greedy()"
