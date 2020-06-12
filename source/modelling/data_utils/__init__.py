from torch.utils.data import DataLoader

from .dataset import EXP, Batch



def get_dataset(data_folder, files, iter_list, cfg): 
    
    dataset = EXP(data_folder, files, iter_list)

    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.SOLVER.BATCH_SIZE, 
        shuffle=True, 
        drop_last=True,
        num_workers=cfg.DATA.NUM_WORKERS, 
        collate_fn=Batch
    )
    return dataloader