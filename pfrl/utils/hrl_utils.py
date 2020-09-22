import numpy as np


def _is_update(episode, freq, ignore=0, rem=0):
    if episode != ignore and episode % freq == rem:
        return True
    return False


def _mean_or_nan(xs):
    """Return its mean a non-empty sequence, numpy.nan for a empty one."""
    return np.mean(xs) if xs else np.nan
