import numpy as np


def concat_obs_and_goal(obs_dict):
    """
    concatenates the observation
    and goal in the observation dictionary.
    """
    return np.concatenate((obs_dict['observation'], obs_dict['desired_goal'])).astype(np.float32)
