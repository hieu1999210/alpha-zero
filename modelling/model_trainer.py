import logging

import numpy as np
import torch
from tqdm import tqdm

from utils import AverageMeter, Timer

from .optimizer_utils import get_lr_scheduler, get_opt


class ModelTrainer:
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

    def __init__(self, model, checkpointer, cfg):

        self.cfg = cfg
        self.model = model
        self.logger = logging.getLogger()
        self.checkpointer = checkpointer
        self.iter = 0
        self.losses = {
            "p": AverageMeter(cache=False),
            "v": AverageMeter(cache=False),
            "loss": AverageMeter(cache=False),
        }
        self.timers = {
            "epoch": Timer(),
            "data": Timer(),
        }
        self.model.train()
        self.model.un_freeze()
        assert self.model.training
        assert not self.model.is_freezed

    def reset(self, dataloader, iter_idx):
        # init optimizer
        optimizer = get_opt(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            self.cfg,
        )

        # init scheduler
        scheduler = get_lr_scheduler(
            optimizer,
            len(dataloader),
            self.cfg,
        )

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.current_epoch = 1
        self.dataloader = dataloader
        self.iter = iter_idx

        self.timers["epoch"].reset()

    def train(self, start_epoch=0):
        """
        NOTE: epoch counter start at 1
        """
        num_epoch = self.cfg.SOLVER.NUM_EPOCHS
        if start_epoch != 0:
            self.current_epoch = start_epoch
        self.timers["epoch"].start()
        while self.current_epoch <= num_epoch:
            self.begin_epoch()
            self.train_loop()
            self.end_epoch()

            self.timers["epoch"].stop()
            self.logger.info(
                f"EPOCH {self.current_epoch:0>3}/{num_epoch:0>3}: "
                f"elapsed time {self.timers['epoch']}, "
                f"p_loss {self.losses['p'].avg:.6f}, "
                f"v_loss {self.losses['v'].avg:.6f}, "
            )
            self.current_epoch += 1
            self.timers["epoch"].start()

    def begin_epoch(self):
        """
        also begin train loop
        """

        # begin train loop
        self.dataset_iter = iter(self.dataloader)
        self.timers["data"].reset()
        for loss in self.losses.values():
            loss.reset()

    def train_loop(self):
        tbar = tqdm(range(len(self.dataloader)))
        for i in tbar:
            self.train_step(i)
            tbar.set_description(
                f"p_loss: {self.losses['p']}, "
                f"v_loss: {self.losses['v']}, "
                f"lr={self.optimizer.param_groups[0]['lr']:.4e}"
            )

    def train_step(self, i):
        """
        i: iter index
        """
        assert self.model.training
        assert not self.model.is_freezed

        # load data
        self.timers["data"].start()
        batch = next(self.dataset_iter)
        batch.cuda()
        self.timers["data"].stop()

        with torch.set_grad_enabled(True):
            # forward
            logits, values = self.model(batch.states)
            loss = self.model.loss((logits, values), (batch.policies, batch.values))
            self.losses["p"].update(loss["pi_loss"].item())
            self.losses["v"].update(loss["v_loss"].item())
            loss = sum(loss.values())
            self.losses["loss"].update(loss.item())

            # backward
            loss /= self.cfg.SOLVER.GD_STEPS
            loss.backward()

            if (i + 1) % self.cfg.SOLVER.GD_STEPS == 0:
                self.scheduler.step()
                self.optimizer.step()
                self.optimizer.zero_grad()

    def end_epoch(self):
        cp_name = (
            f"iter_{self.iter:0>3}_"
            f"epoch_{self.current_epoch:0>3}_"
            f"p_loss_{self.losses['p'].avg:.4f}_"
            f"v_loss_{self.losses['v'].avg:.4f}.pth"
        )
        self.checkpointer.save_checkpoint(
            cp_name,
            epoch=self.current_epoch,
            iter=self.iter,
        )
