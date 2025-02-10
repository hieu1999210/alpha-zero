import logging
import logging.handlers
import os
import sys
import time


def listener_process(queue):
    """
    logging from from the logging queue
    can be run as a proocess or a thread
    """

    while True:
        try:
            record = queue.get()
            # We send this as a sentinel to tell the listener to quit.
            if record is None:
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)  # No level or filter logic applied - just do it!
        except Exception:
            import sys
            import traceback

            print("Whoops! Problem:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


def setup_listen_logger(folder=".", file_name="logs.log", console=True):
    """
    setup logger for the listener process
    """
    root = logging.getLogger()
    formatter = logging.Formatter("{asctime}:  {message}", style="{")
    root.setLevel(logging.INFO)

    if folder:
        assert os.path.isdir(folder), f"log dir '{folder}' does not exist."

        fh = logging.FileHandler(os.path.join(folder, file_name))
        fh.setFormatter(formatter)
        root.addHandler(fh)

    if console:
        ch = logging.StreamHandler(sys.stderr)
        ch.setFormatter(formatter)
        root.addHandler(ch)


def setup_worker_logger(queue):
    """
    setup logger for the worker process
    """
    h = logging.handlers.QueueHandler(queue)  # Just the one handler needed
    root = logging.getLogger()
    root.addHandler(h)
    # send all messages, for demo; no other level or filter logic applied.
    root.setLevel(logging.INFO)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, cache=False, dtype=float):
        self.cache = cache
        self.reset()
        self.dtype = dtype

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.array = [] if self.cache else None

    def update(self, val, n=1):
        if not isinstance(val, self.dtype):
            val = self.dtype(val)

        self.val = val
        if self.cache:
            self.array.append(val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f"{self.val:.4f}({self.avg:.4f})"


class Timer:

    def __init__(self):
        self._timer = AverageMeter(cache=False, dtype=float)

    def reset(self):
        self._timer.reset()

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        elapsed_time = time.perf_counter() - self.start_time
        self._timer.update(elapsed_time)

    def __str__(self):
        return str(self._timer)


def get_log(name, folder=".", file_name="logs.log", console=True):
    """
    return logger for the main
    NOTE: only use in single process run
    """
    assert os.path.isdir(folder), f"log dir '{folder}' does not exist."
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    log_format = logging.Formatter("{asctime}:{name}:  {message}", style="{")
    if folder:
        fh = logging.FileHandler(os.path.join(folder, file_name))
        fh.setFormatter(log_format)
        logger.addHandler(fh)
    # multiprocessing_logging.install_mp_handler()
    if console:
        ch = logging.StreamHandler(sys.stderr)
        ch.setFormatter(log_format)
        logger.addHandler(ch)

    return logger
