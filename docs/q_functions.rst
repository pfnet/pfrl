===========
Q-functions
===========

Q-function interfaces
=====================

.. autoclass:: pfrl.q_function.StateQFunction
   :members:

   .. automethod:: __call__

.. autoclass:: pfrl.q_function.StateActionQFunction
   :members:

   .. automethod:: __call__

Q-function implementations
==========================

.. autoclass:: pfrl.q_functions.DuelingDQN

.. autoclass:: pfrl.q_functions.DistributionalDuelingDQN

.. autoclass:: pfrl.q_functions.SingleModelStateQFunctionWithDiscreteAction

.. autoclass:: pfrl.q_functions.FCStateQFunctionWithDiscreteAction

.. autoclass:: pfrl.q_functions.DistributionalSingleModelStateQFunctionWithDiscreteAction

.. autoclass:: pfrl.q_functions.DistributionalFCStateQFunctionWithDiscreteAction

.. autoclass:: pfrl.q_functions.FCQuadraticStateQFunction

.. autoclass:: pfrl.q_functions.SingleModelStateActionQFunction

.. autoclass:: pfrl.q_functions.FCSAQFunction

.. autoclass:: pfrl.q_functions.FCLSTMSAQFunction

.. autoclass:: pfrl.q_functions.FCBNSAQFunction

.. autoclass:: pfrl.q_functions.FCBNLateActionSAQFunction

.. autoclass:: pfrl.q_functions.FCLateActionSAQFunction
