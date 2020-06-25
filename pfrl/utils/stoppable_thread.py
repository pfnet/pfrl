import threading


class StoppableThread(threading.Thread):
    """Thread with an event object to stop itself.

    Args:
        stop_event (threading.Event): Event that stops the thread if it is set.
        *args, **kwargs: Forwarded to `threading.Thread`.
    """

    def __init__(self, stop_event, *args, **kwargs):
        super(StoppableThread, self).__init__(*args, **kwargs)
        self.stop_event = stop_event

    def stop(self):
        self.stop_event.set()

    def is_stopped(self):
        return self.stop_event.is_set()
