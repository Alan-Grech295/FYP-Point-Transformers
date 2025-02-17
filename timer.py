import time

ENABLED = False

class Timer:
    def __init__(self, message):
        self.start_time = 0
        self.message = message

    def __enter__(self):
        self.start_time = time.perf_counter_ns()

    def __exit__(self, *args):
        if not ENABLED:
            return
        print(f"{self.message} ({(time.perf_counter_ns() - self.start_time) / 1_000_000}ms)")