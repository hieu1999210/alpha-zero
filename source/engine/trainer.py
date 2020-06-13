
import os
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
from .selfplay import Selfplay
from .arena import Arena
from utils import (
    setup_listen_logger,  
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
        print(self.model)     
        self.cfg                = cfg
        self.num_iter           = cfg.SOLVER.NUM_ITERS
        
        self.start_iter         = 1
        self.checkpointer = Checkpointer(
            cfg, logging.getLogger(), self.model
        )
        self.selfplay = Selfplay(self.game, self.model, self.checkpointer, cfg)
        self.arena = Arena(self.game, cfg)
        self.model_trainer = ModelTrainer(self.model, self.checkpointer, cfg)
        
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
                self.timers["selfplay"].start()
                logging.info("\n***SELFPLAY***")
                self.selfplay.plays(_iter)
                self.timers["selfplay"].stop()
                logging.info(f"selfplay time: {self.timers['selfplay']}")
            
            logging.info("\n***TRAIN MODEL***")
            with training_context(self.model):
                self.train_model()
            
            # if _iter >= 2:
            logging.info("\n***COMPARE MODEL***")
            self.timers["compare"].start()
            self.compare_models()
            self.timers["compare"].stop()
            logging.info(f"compare time: {self.timers['compare']}")
            
            self.timers["iter"].stop()
            logging.info(f"iter time: {self.timers['iter']}\n")
            
            self.epoch = 0
            _iter += 1
            self.iter = _iter

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
        
        best_model = self.build_model(self.game, self.cfg)
        self.checkpointer.load_best_cp(best_model)
        best_model_name = self.checkpointer.get_best_cp_name()
        current_model_name = self.checkpointer.get_current_cp_name()
        logging.info(f"{best_model_name} vs {current_model_name}")
        best_model.eval()
        self.model.eval()
        best_model.freeze_param()
        self.model.freeze_param()

        # run matches
        new_wins, old_wins = self.arena.run_matches(self.model, best_model)
        
        logging.info(f"new vs current_best: {new_wins:0>2} - {old_wins:0>2}")
        if (new_wins + old_wins == 0 or 
            float(new_wins)/(new_wins+old_wins) < self.cfg.SOLVER.UPDATE_THRESH
        ):
            logging.info('REJECTING NEW MODEL')
            self.model.load_state_dict(best_model.state_dict())
        else:
            logging.info('ACCEPTING NEW MODEL')
            self.checkpointer.update_best_cp()
                
        del best_model
        torch.cuda.empty_cache()
        
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
