from queue import Empty
import os
import time
import logging
import threading
from contextlib import contextmanager

import torch
import torch.multiprocessing as mp 

from modelling import (
    get_model,
    get_dataset,
    ModelTrainer,
)
from games import build_game
from .selfplay_process import Selfplay
from utils import (
    setup_listen_logger, 
    listener_process, 
    Timer,
    Checkpointer
)


@contextmanager
def training_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    freeze_mode = model.is_freezed
    
    model.train()
    model.un_freeze()
    yield
    model.train(training_mode)
    model.freeze_param(freeze_mode)


class Trainer:
    """
    FLOW:
    for i in range(num_iters):
        selfplay
        train
        compare
        
    """
    
    def __init__(self, cfg):
        
        os.makedirs(cfg.DIRS.EXPERIMENT, exist_ok=True)
        setup_listen_logger(folder=cfg.DIRS.EXPERIMENT)
        
        self.game               = self.build_game(cfg)
        self.model              = self.build_model(self.game, cfg)       
        self.cfg                = cfg
        self.num_iter           = cfg.SOLVER.NUM_ITERS
        self.num_workers        = cfg.SELF_PLAY.NUM_WORKER
        self.games_per_iter     = cfg.SELF_PLAY.GAME_PER_ITER
        self.game_count         = mp.Value('i', 0)
        self.exp_queue          = mp.Queue()
        self.logging_queue      = mp.Queue()
        self.barrier            = mp.Barrier(self.num_workers+1)
        self.start_iter         = 1
        self.checkpointer = Checkpointer(
            cfg, logging.getLogger(), self.model
        )
        self.model_trainer = ModelTrainer(self.model, self.checkpointer, cfg)
        
        self.timers = {
            "selfplay": Timer(),
            "train": Timer(),
            "compare": Timer(),
            "iter": Timer()
        }
        
        exp_path = os.path.join(cfg.DIRS.EXPERIMENT, "training_data")
        os.makedirs(exp_path, exist_ok=True) ############### fix later
        self.exp_folder = exp_path
        
        
        logging.info(f"=========> {cfg.EXPERIMENT} <=========")
        logging.info(f'\n\nStart with config {cfg}')
        
    def run(self):
        """
        run training procedure 
        """
        for idx in range(self.start_iter, self.num_iter+1):
            self.iter = idx
            
            logging.info(f"\n\n******** iter {idx} **********")
            self.timers["iter"].start()
            
            logging.info("\n***SELFPLAY***")
            self.selfplay(idx)
            
            logging.info("\n***TRAIN MODEL***")
            with training_context(self.model):
                self.train_model()
            
            logging.info("\n***COMPARE MODEL***")
            self.compare_models()
            
            self.timers["iter"].stop()
            logging.info(f"iter time: {self.timers['iter']}\n")
    
    def selfplay(self, idx):
        """
        init and run selfplay processes
        gather and save training examples
        terminate selfplay processes
        """
        self.model.eval()
        self.model.freeze_param()
        self.timers["selfplay"].start()
        # init and run worker
        self._start_selfplay()
        
        # wait all workers
        self.barrier.wait()
        
        self.timers["selfplay"].stop()
        logging.info(f"selfplay time: {self.timers['selfplay']}")
        
        # collect sample
        self._collect_samples(idx)
        
        # terminate selfplay process and reset atributes
        self._reset_workers()

    def train_model(self):
        """
        create dataloader
        run model trainer
        """
        self.timers["train"].start()
        dataloader = self._build_dataset()
        self.model_trainer.reset(dataloader, self.iter)
        self.model_trainer.train()
        self.timers["train"].stop()
        logging.info(f"training time: {self.timers['train']}")
        
    def compare_models(self):
        """
        """
        pass
    
    def _start_selfplay(self):
        """
        create and start logging thread and selfplay workers
        """
        
        # create and start logging thread
        self.logging_thread = threading.Thread(
            target=listener_process, 
            args=(self.logging_queue,)
        )
        self.logging_thread.start()
        
        # create and start selfplay workers 
        self.workers = [Selfplay(
            pid=i+1, 
            model=self.model, 
            game=self.game, 
            exp_queue=self.exp_queue, 
            logging_queue=self.logging_queue,
            global_games_count=self.game_count,
            barrier=self.barrier,
            total_num_games=self.games_per_iter, 
            cfg=self.cfg,
        ) for i in range(self.num_workers)]

        for worker in self.workers:
            worker.start()

    def _collect_samples(self, idx):
        """
        collect and save samples generated by selfplaying
        format: {
            "states": torch tensor of shape (N, 1, H, W) 
                        where H,W is board size
            "policies": torch tensor of shape (N, V)
                        where V is number of moves
            "values": torch tensor of shape (N,1)
        }
        """
        
        num_exp = self.exp_queue.qsize()
        logging.info(f"collecting {num_exp} samples")
        
        x,y = self.game.getBoardSize()
        num_moves = self.game.getActionSize()
        
        boards = torch.zeros((num_exp, 1, x, y))
        policies = torch.zeros((num_exp, num_moves))
        values = torch.zeros((num_exp,1))
        for i in range(num_exp):
            board, policy, value = self.exp_queue.get()
            boards[i,0] = torch.from_numpy(board)
            policies[i] = torch.from_numpy(policy)
            values[i,0] = value
        
        exp_file = os.path.join(self.exp_folder, f"iter_{idx:0>3}_{num_exp:0>6}_samples.pth")
        data = {
            "states": boards.type(torch.int8),
            "policies": policies,
            "values": values
        }
        
        self.checkpointer.save_data(exp_file, data, iter=self.iter)
        
        del boards
        del policies
        del values
    
    def _reset_workers(self):
        """
        terminate workers and reset attributes
        """
        
        # finish workers
        for worker in self.workers:
            worker.join()
        
        # finish logging thread
        self.logging_queue.put_nowait(None)
        self.logging_thread.join()
        
        # reset attributes
        self.game_count = mp.Value("i", 0)
        self.exp_queue = mp.Queue()
        self.barrier = mp.Barrier(self.num_workers + 1)
        self.logging_queue = mp.Queue()
    
    def _build_dataset(self):
        history_length = self.cfg.DATA.HISTORY_LENGTH
        current_iter = self.iter
        
        iter_list = []
        for _ in range(history_length):
            if current_iter > 0:
                iter_list.append(current_iter)
                current_iter -= 1
                
        return get_dataset(
            self.checkpointer.data_folder, 
            self.checkpointer.logs["train_data"],
            iter_list,
            self.cfg
        )
    @classmethod
    def build_game(cls, cfg):
        return build_game(cfg)
    
    @classmethod
    def build_model(cls, game, cfg):
        model = get_model(game, cfg)
        model.eval()
        model.freeze_param()
        model.cuda().share_memory()
        
        return model