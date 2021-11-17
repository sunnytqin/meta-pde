"""Simple class to time execution"""
import time


class Timer:
    """Use Timer in a with statement to time execution

    e.g. given a method method_to_time()...

    with Timer() as t:
        outputs = method_to_time(inputs)

    print("Time taken: {}".format(t.interval))
    """

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start


def time_fn(fn, *args, **kwargs):
    with Timer() as t:
        fn(*args, **kwargs)
    return t.interval
