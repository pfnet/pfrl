from abc import ABCMeta, abstractmethod

# Delay importing optuna since it is an optional dependency.
try:
    import optuna

    _optuna_available = True
except ImportError:
    _optuna_available = False


class EvaluationHook(object, metaclass=ABCMeta):
    """Hook function that will be called after evaluation.

    This class is for clarifying the interface required for EvaluationHook functions.
    You don't need to inherit this class to define your own hooks. Any callable that
    accepts (env, agent, evaluator, step, eval_score) as arguments can be used as an
    evaluation hook.

    Note that:
    - ``step`` is the current training step, not the number of evaluations so far.
    - ``train_agent_async`` DOES NOT support EvaluationHook.
    """

    @abstractmethod
    def __call__(self, env, agent, evaluator, step, eval_score):
        """Call the hook.

        Args:
            env: Environment.
            agent: Agent.
            evaluator: Evaluator.
            step: Current timestep.
            eval_score: Evaluation score at t=`step`.
        """
        raise NotImplementedError


class OptunaPrunerHook(EvaluationHook):
    """Report evaluation scores to Optuna and prune the trial if necessary.

    Optuna regards trials which raise `optuna.TrialPruned` as unpromissed and
    prune them at the early stages of the training.

    Note that:
    - ``step`` is the current training step, not the number of evaluations so far.
    - ``train_agent_async`` DOES NOT support EvaluationHook.
      - This hook stops trial by raising an exception, but re-raise error among process
        is not straight forward.

    Args:
        trial (optuna.Trial): Current trial.
    Raises:
        optuna.TrialPruned: Raise when the trial should be pruned immediately.
            Note that you don't need to care about this exception since Optuna will
            catch `optuna.TrialPruned` and stop the trial properly.
    """

    def __init__(self, trial):
        if not _optuna_available:
            raise RuntimeError("OptunaPrunerHook requires optuna installed.")
        self.trial = trial

    def __call__(self, env, agent, evaluator, step, eval_score):
        self.trial.report(eval_score, step)
        if self.trial.should_prune():
            raise optuna.TrialPruned()
