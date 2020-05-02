import time
import os
import csv
from contextlib import contextmanager

import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch


from .monitor import Monitor
from .my_evaluator import Evaluator
from projects.segmentation.optimizer_utils import get_opt, get_lr_scheduler
from projects.segmentation.data_utils import get_dataset, get_testdataset
from .base_classes import BaseTrainer
from projects.segmentation.utils import (
    get_log, write_log, write_stats, AverageMeter, Checkpointer
)

@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
    
    
class Trainer(BaseTrainer):
    """
    trainer will run in following procedure:
    train():
        for e in epoch:
            begin_epoch()
            train_loop():
                train_step()
            after_train_loop()
            validate()
            end_epoch() ## detectron combine this step to after_train
    """
    def __init__(self, cfg, args):
        
        # check if overive previous experiments
        if not args.resume:
            assert not os.path.isdir(cfg.DIRS.EXPERIMENT),\
                "{} exists".format(cfg.DIRS.EXPERIMENT)
            os.makedirs(cfg.DIRS.EXPERIMENT)
        self.cfg = cfg
        
        # make directory to store training stats
        stats_folder = os.path.join(cfg.DIRS.EXPERIMENT, 'stats')
        if not os.path.isdir(stats_folder):
            os.makedirs(stats_folder)
        self._stats_folder = stats_folder
        
        # init global logger
        logger = get_log("main", cfg.DIRS.EXPERIMENT)
        self.logger = logger
        logger.info(f"=========> {cfg.EXPERIMENT} <=========")
        logger.info(f'\n\nStart with config {cfg}')
        logger.info(f'Command arguments {args}')
        
        # init dataloader
        train_dataloader = self.build_dataloader(cfg, "train", logger)
        val_dataloader = self.build_dataloader(cfg, "val", logger)
        self.train_dataloader = train_dataloader
        
        # init model
        model = self.build_model(cfg, logger)
        
        # init optimizer
        optimizer = get_opt(filter(lambda p: p.requires_grad, model.parameters()), cfg)
        
        # using apex ### using torch.cuda.amp instead
        # model, optimizer = amp.initialize(
        #     model, 
        #     optimizer, 
        #     opt_level=cfg.SYSTEM.OPT_L,
        #     keep_batchnorm_fp32=True, 
        #     loss_scale="dynamic"
        # )
        self.model = model
        self.optimizer = optimizer
        self.scaler = GradScaler()
        
        # init scheduler
        scheduler = get_lr_scheduler(optimizer, len(train_dataloader), cfg)
        self.scheduler = scheduler
        
        # tensorboard
        tb = SummaryWriter(os.path.join(self.cfg.DIRS.EXPERIMENT, "tb"))
        self.tensorboard = tb
        
        # init monitor
        monitor = Monitor(loss_names=["seg"])
        self.monitor = monitor
        
        # init checkpointer
        self.checkpointer = Checkpointer(
            cfg=cfg,
            logger=logger, 
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        self.current_epoch = self.checkpointer.load_resume(args.load)
        # print("################", self.current_epoch)
        # init evaluator
        self.evaluator = Evaluator(
            val_dataloader=val_dataloader,
            monitor=monitor,
            model=model,
            tb=tb,
            logger=logger,
            current_epoch=self.current_epoch,
            cfg=cfg,
        )
        
        # init epoch time meter
        self.epoch_timer = AverageMeter(cache=False)
    
    def train(self):
        num_epoch = self.cfg.SOLVER.NUM_EPOCHS
        end_epoch = time.perf_counter()
        self.model.train()
        # print("####################", self.current_epoch)
        while self.current_epoch <= num_epoch:
            self.begin_epoch()
            self.train_loop()
            self.after_train_loop()
            results = self.validate()
            self.end_epoch(results)
            
            self.epoch_timer.update(time.perf_counter() - end_epoch)
            end_epoch = time.perf_counter()
            self.logger.info(
                f'epoch: [{(self.current_epoch):0>3}/{num_epoch:0>3}] \t'
                f'Overall Time {self.epoch_timer} \t'
            )
            self.current_epoch += 1
    
    def begin_epoch(self):
        """
        also begin train loop
        """
        epoch = self.current_epoch
        self.current_epoch = epoch
        self.logger.info(f'\n\n*****\nEPOCH {epoch:0>3}/{self.cfg.SOLVER.NUM_EPOCHS:0>3}:')
        self.logger.info('TRAIN PHASE:')
        
        # get stats writer
        stats_path = os.path.join(self._stats_folder, f'epoch_{epoch+1:0>3}.csv')
        stats_file = open(stats_path, 'w', newline='')
        csv_writer = csv.writer(stats_file)
        self._start_file = stats_file
        self._csv_writer = csv_writer

        # begin train loop
        self.train_iter = iter(self.train_dataloader)
        self.monitor.reset()
    
    def train_loop(self):
        tbar = tqdm(range(len(self.train_dataloader)))
        for i in tbar:
            self.train_step(i)
            tbar.set_description(
                self.monitor.to_str() +
                f', lr={self.optimizer.param_groups[0]["lr"]:.4e}'
            )

    def train_step(self, i):
        """
            i: iter index
        """
        assert self.model.training
        
        # load data
        start = time.perf_counter()
        batch = next(self.train_iter)
        batch.cuda()
        data_time = time.perf_counter() - start
        self.monitor.update_time(data_time)
        
        inputs = batch.images
        gts = batch.targets

        with torch.set_grad_enabled(True):
            # forward
            with autocast():
                logits, loss = self.model(inputs, gts)
            

            # backward
            loss /= self.cfg.SOLVER.GD_STEPS
            self.scaler.scale(loss).backward()
            
            if (i+1) % self.cfg.SOLVER.GD_STEPS == 0:
                self.scheduler.step()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        
        # logging
        with torch.no_grad():
            pred = self.model.infer(logits)
            self.monitor.update_metric(pred, gts)
            
            loss = loss*self.cfg.SOLVER.GD_STEPS
            self.monitor.update_loss(seg=loss)

            log = self.monitor.get_loss_val()
            log.update(self.monitor.get_metric_val())
            log["lr"] = self.optimizer.param_groups[0]["lr"]
            
            self.tensorboard.add_scalars(
                "train",
                log,
                (self.current_epoch-1)*len(self.train_dataloader) + i   
            )
    
    def after_train_loop(self):
        """
        evaluating training loop and write out logs
        """
        self.monitor.eval()
        self._save_results(mode="train")
    
    def validate(self):
        self.evaluator.run_eval()
        results = self._save_results(mode="val")
        return results
    
    def end_epoch(self, val_result):
        self._start_file.close()
        
        cp_name = "{}_epoch_{:0>3}_iou_{:.4f}.pth".format(
            self.cfg.MODEL.META_ARCHITECTURE,
            self.current_epoch, 
            val_result["iou"],
        )
        self.checkpointer.save_checkpoint(
            cp_name, 
            epoch=self.current_epoch, 
            current_metric=val_result[self.cfg.SOLVER.MAIN_METRIC])

    def _save_results(self, mode):
        """
        write results to logging
        args: 
            mode: "val" or "train"
        return results
        """
        assert mode in ["val", "train"], "invalid mode"
        
        results = self.monitor.results
        results.update(self.monitor.get_mean_loss())
        
        self.tensorboard.add_scalars(f"{mode}_mean", results, self.current_epoch)
        write_log(self.logger.info, mode=mode, **results)
        write_stats(self._csv_writer, mode=mode, 
            **(self.monitor.get_loss_array()),
            **results,
        )
        
        return results
    
    @classmethod
    def build_model(cls, cfg, logger=None):
        if logger:
            logger.info('loading model')
        model = build_model(cfg).cuda()
        return model
    
    @classmethod
    def build_dataloader(cls, cfg, mode, logger=None):
        assert mode in ["test", "val", "train"]
        if logger:
            logger.info(f'loading {mode} dataset')
        
        if mode == "test":
            dataloader = get_testdataset(cfg)
        else:
            dataloader = get_dataset(mode, cfg)
        
        return dataloader
