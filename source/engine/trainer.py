from queue import Empty
import os
import time
import logging
import threading
from contextlib import contextmanager

import torch
import numpy as np
import torch.multiprocessing as mp 

from modelling import (
    get_model,
    get_dataset,
    ModelTrainer,
)
from games import build_game
from .selfplay_process import Selfplay
from .matches_process import Match
from .mcts import MCTS
from .Arena import Arena
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
    is_freezed = model.is_freezed
    
    model.train()
    model.un_freeze()
    yield
    model.train(training_mode)
    if is_freezed:
        model.freeze_param()
    else:
        model.un_freeze()


class Trainer:
    """
    FLOW:
    for i in range(num_iters):
        selfplay
        train
        compare
        
    """
    
    def __init__(self, cfg, resume=False):
        
        if not resume:
            _dir = cfg.DIRS.EXPERIMENT
            # assert not os.path.exists(_dir), f"{_dir} exits"
            os.makedirs(_dir)
        setup_listen_logger(folder=cfg.DIRS.EXPERIMENT)
        
        self.game               = self.build_game(cfg)
        self.model              = self.build_model(self.game, cfg)       
        self.cfg                = cfg
        self.num_iter           = cfg.SOLVER.NUM_ITERS
        
        ### atributes for selfplay
        self.num_self_workers   = cfg.SELF_PLAY.NUM_WORKER
        self.game_count         = mp.Value('i', 0)
        self.exp_queue          = mp.Queue()
        self.self_logging_queue = mp.Queue()
        self.self_barrier       = mp.Barrier(self.num_self_workers+1)
        self.self1_wins_count   = mp.Value("i", 0)
        self.self2_wins_count   = mp.Value("i", 0)
        
        ### atributes for compare models
        self.num_match_workers  = cfg.MATCH.NUM_WORKERS
        self.match_barrier      = mp.Barrier(self.num_match_workers+1)
        self.matches_count      = mp.Value('i', 0)
        self.match_log_queue    = mp.Queue()
        # count for new model
        self.new_wins_count     = mp.Value("i", 0)
        # count for best old model
        self.old_wins_count     = mp.Value("i", 0)
        
        self.start_iter         = 1
        self.checkpointer = Checkpointer(
            cfg, logging.getLogger(), self.model
        )
        self.model_trainer = ModelTrainer(self.model, self.checkpointer, cfg)
        
        
        
        exp_path = os.path.join(cfg.DIRS.EXPERIMENT, "training_data")
        os.makedirs(exp_path, exist_ok=True) ############### fix later
        self.exp_folder = exp_path
        
        logging.info(f"=========> {cfg.EXPERIMENT} <=========")
        logging.info(f'\n\nStart with config {cfg}')
        self.epoch, self.iter = self.checkpointer.load_resume()
        self.timers = {
            "selfplay": Timer(),
            "train": Timer(),
            "compare": Timer(),
            "iter": Timer()
        }
        
    def run(self):
        """
        run training procedure 
        
        NOTE: in a iter if current epoch >0 selfplay is skipped
        
        """
        _iter = self.iter
        while _iter <= self.num_iter:
            
            logging.info(f"\n\n******** iter {_iter} **********")
            self.timers["iter"].start()
            
            if self.epoch == 0:
                logging.info("\n***SELFPLAY***")
                self.selfplay(_iter)
            
            logging.info("\n***TRAIN MODEL***")
            with training_context(self.model):
                self.train_model()
            
            # if _iter >= 2:
            logging.info("\n***COMPARE MODEL***")
            self.compare_models()
            
            self.timers["iter"].stop()
            logging.info(f"iter time: {self.timers['iter']}\n")
            
            self.epoch = 0
            _iter += 1
            self.iter = _iter
    
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
        self.self_barrier.wait()
        
        self.timers["selfplay"].stop()
        logging.info(f"player1 wins: {self.self1_wins_count.value:0>3}  "
                     f"player2 wins: {self.self2_wins_count.value:0>3}")
        logging.info(f"selfplay time: {self.timers['selfplay']}")
        
        # collect sample
        self._collect_samples(idx)
        
        # terminate selfplay process and reset atributes
        self._reset_selfplay()
        # torch.cuda.empty_cache()
        
    def train_model(self):
        """
        create dataloader
        run model trainer
        """
        self.timers["train"].start()
        dataloader = self._build_dataset()
        self.model_trainer.reset(dataloader, self.iter)
        self.model_trainer.train(self.epoch)
        self.timers["train"].stop()
        logging.info(f"training time: {self.timers['train']}")

    def compare_models(self):
        self.timers["compare"].start()
        best_model = self.build_model(self.game, self.cfg)
        self.checkpointer.load_best_cp(best_model)
        
        best_model.eval()
        self.model.eval()
        best_model.freeze_param()
        self.model.freeze_param()

        # init and run worker
        self._start_matches(best_model)
        
        # wait all workers
        self.match_barrier.wait()

        new_wins = self.new_wins_count.value
        old_wins = self.old_wins_count.value
        logging.info(f"NEW/PREV WINS : {new_wins:0>2} / {old_wins:0>2}")

        if (new_wins + old_wins == 0 or 
            float(new_wins) / (new_wins + old_wins) < self.cfg.SOLVER.UPDATE_THRESH):
            logging.info('REJECTING NEW MODEL')
            self.model.load_state_dict(best_model.state_dict())
        else:
            logging.info('ACCEPTING NEW MODEL')
            self.checkpointer.update_best_cp()
        
        # terminate selfplay process and reset atributes
        self._reset_match_workers()
        
        del best_model
        torch.cuda.empty_cache()
        self.timers["compare"].stop()
        logging.info(f"compare time: {self.timers['compare']}")
    
    def _start_selfplay(self):
        """
        create and start logging thread and selfplay workers
        """
        
        # create and start logging thread
        self.self_logging_thread = threading.Thread(
            target=listener_process, 
            args=(self.self_logging_queue,)
        )
        self.self_logging_thread.start()
        
        # create and start selfplay workers 
        self.self_workers = [Selfplay(
            pid=i+1, 
            model=self.model, 
            game=self.game, 
            exp_queue=self.exp_queue, 
            logging_queue=self.self_logging_queue,
            global_games_count=self.game_count,
            barrier=self.self_barrier,
            cfg=self.cfg,
            p1_wins=self.self1_wins_count,
            p2_wins=self.self2_wins_count,
        ) for i in range(self.num_self_workers)]

        for worker in self.self_workers:
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
        
        self.checkpointer.save_data(exp_file, data, iter=self.iter, epoch=0)
        
        del boards
        del policies
        del values
    
    def _reset_selfplay(self):
        """
        terminate workers and reset attributes
        """
        
        # finish workers
        for worker in self.self_workers:
            worker.join()
        
        # finish logging thread
        self.self_logging_queue.put_nowait(None)
        self.self_logging_thread.join()
        
        # reset attributes
        self.game_count = mp.Value("i", 0)
        self.self1_wins_count = mp.Value("i", 0)
        self.self2_wins_count = mp.Value("i", 0)
        self.exp_queue = mp.Queue()
        
        self.self_barrier = mp.Barrier(self.num_self_workers + 1)
        self.self_logging_queue = mp.Queue()
    
    def _start_matches(self, old_model):
        """
        create and start logging thread and match workers
        """
        
        # create and start logging thread
        self.match_logging_thread = threading.Thread(
            target=listener_process, 
            args=(self.match_log_queue,)
        )
        self.match_logging_thread.start()
        # print("########## num match workers", self.num_match_workers)
        # create and start selfplay workers 
        self.match_workers = [Match(
            pid=i+1, 
            model1=self.model, 
            p1_wins=self.new_wins_count,
            model2=old_model,
            p2_wins=self.old_wins_count,
            game=self.game, 
            logging_queue=self.match_log_queue,
            global_match_count=self.matches_count,
            barrier=self.match_barrier,
            cfg=self.cfg,
        ) for i in range(self.num_match_workers)]

        for worker in self.match_workers:
            worker.start()
    
    def _reset_match_workers(self):
        """
        reset match_workers and variables
        """
        for worker in self.match_workers:
            worker.join()
        
        # finish logging thread
        self.match_log_queue.put_nowait(None)
        self.match_logging_thread.join()
        
        # reset attributes
        self.matches_count = mp.Value("i", 0)
        self.old_wins_count = mp.Value("i", 0)
        self.new_wins_count = mp.Value("i", 0)
        
        self.match_barrier = mp.Barrier(self.num_match_workers + 1)
        self.match_log_queue = mp.Queue()
    
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
    
    # def compare_models(self):
    #     """
    #     """
    #     self.timers["compare"].start()
    #     best_model = self.build_model(self.game, self.cfg)
    #     self.checkpointer.load_best_cp(best_model)
    #     self.model.eval()
    #     best_model.eval()
    #     best_player = MCTS(self.game, self.cfg, best_model)
        
    #     current_player = MCTS(self.game, self.cfg, self.model)
        
    #     arena = Arena(
    #         lambda x: np.argmax(current_player.get_policy_infer(x, temp=0)),
    #         lambda x: np.argmax(best_player.get_policy_infer(x, temp=0)),
    #         self.game,
    #     )
    #     curr_wins, best_wins, draws = arena.playGames(
    #         self.cfg.MATCH.NUM_MATCHES)
    #     self.timers["compare"].stop()
    #     logging.info(
    #         f'NEW/PREV WINS : {curr_wins:0>2} / {best_wins:0>2} ;' + 
    #         f' DRAWS : {draws:0>2}')
    #     if (curr_wins + best_wins == 0 or 
    #         float(curr_wins) / (curr_wins + best_wins) < self.cfg.SOLVER.UPDATE_THRESH):
    #         logging.info('REJECTING NEW MODEL')
    #         self.model.load_state_dict(best_model.state_dict())
    #     else:
    #         logging.info('ACCEPTING NEW MODEL')
    #         self.checkpointer.update_best_cp()
        
    #     del best_model
    #     torch.cuda.empty_cache()