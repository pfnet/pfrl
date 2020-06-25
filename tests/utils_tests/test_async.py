import multiprocessing as mp
import os
import signal
import sys
import unittest
import warnings

from pfrl.utils import async_


class TestAsync(unittest.TestCase):
    def test_run_async(self):
        counter = mp.Value("l", 0)

        def run_func(process_idx):
            for _ in range(1000):
                with counter.get_lock():
                    counter.value += 1

        async_.run_async(4, run_func)
        self.assertEqual(counter.value, 4000)

    def test_run_async_exit_code(self):
        def run_with_exit_code_0(process_idx):
            sys.exit(0)

        def run_with_exit_code_11(process_idx):
            os.kill(os.getpid(), signal.SIGSEGV)

        with warnings.catch_warnings(record=True) as ws:
            async_.run_async(4, run_with_exit_code_0)
            # There should be no AbnormalExitWarning
            self.assertEqual(
                sum(
                    1 if issubclass(w.category, async_.AbnormalExitWarning) else 0
                    for w in ws
                ),
                0,
            )

        with warnings.catch_warnings(record=True) as ws:
            async_.run_async(4, run_with_exit_code_11)
            # There should be 4 AbnormalExitWarning
            self.assertEqual(
                sum(
                    1 if issubclass(w.category, async_.AbnormalExitWarning) else 0
                    for w in ws
                ),
                4,
            )
