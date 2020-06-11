from torch.multiprocessing import set_start_method
from engine import Trainer
from utils import (
    get_cfg_defaults, 
    parse_args, 
    setup_config,
)


def main(cfg , args):
    if args.mode == "train":
        trainer = Trainer(cfg, resume=args.resume)
        trainer.run()
    else:
        raise ValueError("Invalid mode")
    
if __name__ == "__main__":
    cfg = get_cfg_defaults()
    args = parse_args()
    cfg = setup_config(cfg, args)
    
    set_start_method('spawn', force=True)
    main(cfg, args)