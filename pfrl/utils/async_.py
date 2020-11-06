import warnings

import torch.multiprocessing as mp


class AbnormalExitWarning(Warning):
    """Warning category for abnormal subprocess exit."""

    pass


def run_async(n_process, run_func):
    """Run experiments asynchronously.

    Args:
      n_process (int): number of processes
      run_func: function that will be run in parallel
    """

    processes = []

    for process_idx in range(n_process):
        processes.append(mp.Process(target=run_func, args=(process_idx,)))

    for p in processes:
        p.start()

    for process_idx, p in enumerate(processes):
        p.join()
        if p.exitcode > 0:
            warnings.warn(
                "Process #{} (pid={}) exited with nonzero status {}".format(
                    process_idx, p.pid, p.exitcode
                ),
                category=AbnormalExitWarning,
            )
        elif p.exitcode < 0:
            warnings.warn(
                "Process #{} (pid={}) was terminated by signal {}".format(
                    process_idx, p.pid, -p.exitcode
                ),
                category=AbnormalExitWarning,
            )
