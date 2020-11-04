import threading
import unittest

from pfrl.utils import StoppableThread


class TestStoppableThread(unittest.TestCase):
    def test_stoppable_thread(self):
        stop_event = threading.Event()
        thread = StoppableThread(stop_event=stop_event)
        self.assertFalse(thread.is_stopped())
        thread.stop()
        self.assertTrue(stop_event.is_set())
        self.assertTrue(thread.is_stopped())
