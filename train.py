from torch.multiprocessing import set_start_method

from engine import Trainer
from utils import get_cfg_defaults, parse_args, set_deterministic, setup_config


def main(cfg, args):
    trainer = Trainer(cfg, resume=args.resume)
    trainer.run()


if __name__ == "__main__":
    cfg = get_cfg_defaults()
    args = parse_args()
    cfg = setup_config(cfg, args)

    set_start_method("spawn", force=True)
    set_deterministic()
    main(cfg, args)
