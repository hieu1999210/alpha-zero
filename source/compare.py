from engine import Arena, Trainer
from games import build_game
from utils import get_cfg_defaults, set_deterministic, setup_listen_logger
from modelling import get_model
import torch
from torch.multiprocessing import set_start_method
import logging
import os
# cp_folder = "/home/hieu123/alpha_zero_mp/runs/exp4_hust/checkpoints"
cp_folder = "/home/hieu123/alpha_zero_mp2/checkpoints"

def run():
    setup_listen_logger(file_name="compare.log")
    cfg = get_cfg_defaults()
    # cfg_path = ("/home/hieu123/alpha_zero_mp/configs/exp4_hust.yaml")
    cfg_path = ("/home/hieu123/alpha_zero_mp2/configs/exp2_local.yaml")
    cfg.merge_from_file(cfg_path)
    cfg.freeze()
    print(cfg)
    game = build_game(cfg)

    # model1
    # cp_path1 = os.path.join(cp_folder, "iter_183_epoch_010_p_loss_1.2921_v_loss_0.5793.pth")
    cp_path1 = os.path.join(cp_folder, "exp2_local_iter_195_epoch_010_p_loss_0.4580_v_loss_0.1726.pth")
    model1 = get_model(game, cfg)
    cp1 = torch.load(cp_path1, map_location="cpu")
    model1.load_state_dict(cp1["state_dict"])
    model1.eval()
    model1.freeze_param()
    model1.cuda().share_memory()
    
    # model2
    cfg_path2 = "/home/hieu123/alpha_zero_mp2/configs/exp2_local.yaml"
    cfg2 = get_cfg_defaults()
    cfg2.merge_from_file(cfg_path2)
    # cfg2 = cfg
    # cp_path2 = os.path.join(cp_folder, "exp4_hust_iter_183_epoch_010_p_loss_1.2921_v_loss_0.5793.pth")
    cp_path2 = os.path.join(cp_folder, "8x8_100checkpoints_best.pth.tar")
    model2 = get_model(game, cfg2)
    cp2 = torch.load(cp_path2, map_location="cpu")
    model2.load_state_dict(cp2["state_dict"])
    model2.eval()
    model2.freeze_param()
    model2.cuda().share_memory()

    arena = Arena(game, cfg, cfg2=cfg2)
    logging.info(arena.run_matches(model1, model2))

if __name__ == "__main__":
    set_deterministic()
    set_start_method('spawn', force=True)
    run()