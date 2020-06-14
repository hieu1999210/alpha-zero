import torch

from .lr_scheduler import LR_Scheduler


def get_opt(weights, cfg):
    return torch.optim.AdamW(
        weights, 
        lr=cfg.SOLVER.BASE_LR, 
        weight_decay=cfg.SOLVER.WEIGHT_DECAY
    )
    

def get_lr_scheduler(optimizer, num_batches, cfg):
    """
    args:
        iter_num: len(dataloader)
        
    return [warmup_scheduler, annualing_scheduler]
    """
    scheduler = LR_Scheduler(
        optimizer=optimizer,
        mode="cos",
        base_lr=cfg.SOLVER.BASE_LR,
        num_epochs=cfg.SOLVER.NUM_EPOCHS,
        iters_per_epoch=num_batches//cfg.SOLVER.GD_STEPS,
        min_lr=cfg.SOLVER.BASE_LR/10,
        warmup_epochs=cfg.SOLVER.WARMUP_EPOCHS,
    )
    
    # warmup_scheduler = WarmUpLR(
    #     optimizer, 
    #     num_iter/cfg.OPT.GD_STEPS)
    
    return scheduler