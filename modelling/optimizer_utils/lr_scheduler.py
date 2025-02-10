import math


class LR_Scheduler(object):
    """Learning Rate Scheduler

    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``

    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``

    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``

    Args:
        args:  :attr:`args.lr_scheduler` lr scheduler mode (`cos`, `poly`),
          :attr:`args.lr` base learning rate, :attr:`args.epochs` number of epochs,
          :attr:`args.lr_step`

        iters_per_epoch: number of iterations per epoch
    """

    def __init__(
        self,
        optimizer,
        mode,
        base_lr,
        num_epochs,
        iters_per_epoch=0,
        min_lr=1e-5,
        lr_step=0,
        warmup_epochs=0,
    ):
        self.mode = mode
        print("Using {} LR Scheduler!".format(self.mode))
        self.base_lr = base_lr
        self.min_lr = min_lr
        if mode == "step":
            assert lr_step
        self.lr_step = lr_step
        self.iters_per_epoch = iters_per_epoch
        self.N = num_epochs * iters_per_epoch
        # self.epoch = -1
        self.warmup_iters = warmup_epochs * iters_per_epoch
        self._param_groups = optimizer.param_groups
        self.current_step = 0

    def state_dict(self):
        return {
            "current_step": self.current_step,
        }

    def load_state_dict(self, state_dict):
        for key, value in state_dict.items():
            setattr(self, key, value)

    def step(self):
        T = self.current_step
        epoch = T // self.iters_per_epoch

        if T < self.warmup_iters:
            lr = self.base_lr * 1.0 * T / self.warmup_iters
        elif self.mode == "cos":
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + math.cos(1.0 * T / self.N * math.pi)
            )
        elif self.mode == "poly":
            lr = self.base_lr * pow((1 - 1.0 * T / self.N), 0.9)
        elif self.mode == "step":
            lr = self.base_lr * (0.1 ** (epoch // self.lr_step))
        else:
            raise NotImplemented
        # if epoch > self.epoch:
        # print('\n=>Epoches %i, learning rate = %.4f, \
        #     previous best = %.4f' % (epoch, lr, best_pred))
        # self.epoch = epoch
        assert lr >= 0
        self._adjust_learning_rate(lr)
        self.current_step += 1

    def _adjust_learning_rate(self, lr):
        for param_group in self._param_groups:
            param_group["lr"] = lr


# class WarmUpLR(_LRScheduler):
#     """warmup_training learning rate scheduler
#     Args:
#         optimizer: optimzier(e.g. SGD)
#         total_iters: totoal_iters of warmup phase
#     """
#     def __init__(self, optimizer, total_iters, last_epoch=-1):

#         self.total_iters = total_iters
#         super().__init__(optimizer, last_epoch)

#     def get_lr(self):
#         """we will use the first m batches, and set the learning
#         rate to base_lr * m / total_iters
#         """
#         return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
