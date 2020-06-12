"""
pretrained: ../pretrained_models/exp2_local_iter_037_epoch_010_p_loss_0.5376_v_loss_0.1458.pth
cfg: ./configs/exp2_local_demo.yaml
"""


from tkinter import Tk, Canvas
import torch
import numpy as np
from engine import MCTS, GUI, ArenaGUI
from games import build_game
from modelling import get_model
from utils import get_log, get_cfg_defaults, setup_config, parse_args, COLOR


def run_gui(cfg, args):
    
    root = Tk()
    screen = Canvas(
        root, width=cfg.GUI.WIDTH, height=cfg.GUI.HEIGHT, 
        background=COLOR[0], highlightthickness=0
    )
    
    screen.pack()
    game = build_game(cfg)
    logger = get_log(name="demo", file_name="demo.log")
    # init player 

    
    #
    model = get_model(game, cfg)
    cp = torch.load(args.load, map_location="cpu")
    model.load_state_dict(cp["state_dict"])
    model.eval()
    model.freeze_param()
    model.to(cfg.DEVICE)
    mcts = MCTS(game, cfg, model) 

    AI_player = lambda x: np.argmax(mcts.get_policy_infer(x, temp=0))

    gui = GUI(root, screen, logger, COLOR, cfg)
    arena = ArenaGUI(AI_player, game, gui.update_state)
    
    gui.start(arena=arena)
    #Run forever
    root.wm_title("Othello - developed by nhom7")
    root.mainloop()

if __name__ == "__main__":
    args = parse_args()
    cfg = get_cfg_defaults()
    cfg = setup_config(cfg, args)
    run_gui(cfg, args)