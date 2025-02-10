import threading

import torch.multiprocessing as mp

from utils import listener_process

from .match_worker import MatchWorker


class Arena:

    def __init__(self, game, cfg, cfg2=None):

        # fmt: off
        self.game               = game 
        self.cfg                = cfg
        self.cfg2               = cfg2
        self.num_workers        = cfg.MATCH.NUM_WORKERS
        self.barrier            = mp.Barrier(self.num_workers+1)
        self.matches_count      = mp.Value('i', 0)
        self.log_queue          = mp.Queue()
        self.player1_wins_count = mp.Value("i", 0)
        self.player2_wins_count = mp.Value("i", 0)
        self.update_thresh      = self.cfg.SOLVER.UPDATE_THRESH
        # fmt: on

    def run_matches(self, model1, model2):

        # init and run worker
        self._start_workers(model1, model2)

        # wait all workers
        self.barrier.wait()

        wins1 = self.player1_wins_count.value
        wins2 = self.player2_wins_count.value
        self._reset_workers()

        return wins1, wins2

    def _start_workers(self, model1, model2):
        """
        create and start logging thread and match workers
        """

        # create and start logging thread
        self.logging_thread = threading.Thread(
            target=listener_process, args=(self.log_queue,)
        )
        self.logging_thread.start()
        # print("########## num match workers", self.num_workers)
        # create and start selfplay workers
        self.workers = [
            MatchWorker(
                pid=i + 1,
                model1=model1,
                p1_wins=self.player1_wins_count,
                model2=model2,
                p2_wins=self.player2_wins_count,
                game=self.game,
                logging_queue=self.log_queue,
                global_match_count=self.matches_count,
                barrier=self.barrier,
                cfg=self.cfg,
                cfg2=self.cfg2,
            )
            for i in range(self.num_workers)
        ]

        for worker in self.workers:
            worker.start()

    def _reset_workers(self):
        """
        reset workers and variables
        """
        for worker in self.workers:
            worker.join()

        # finish logging thread
        self.log_queue.put_nowait(None)
        self.logging_thread.join()

        # reset attributes
        self.matches_count = mp.Value("i", 0)
        self.player2_wins_count = mp.Value("i", 0)
        self.player1_wins_count = mp.Value("i", 0)

        self.barrier = mp.Barrier(self.num_workers + 1)
        self.log_queue = mp.Queue()
