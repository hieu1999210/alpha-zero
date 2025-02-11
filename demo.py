from tkinter import Canvas, Tk

import numpy as np
import torch

from engine import GUI, MCTS, ArenaGUI
from games import build_game
from modelling import get_model
from utils import COLOR, get_cfg_defaults, get_log, parse_args, setup_config


def run_gui(cfg):
    game = build_game(cfg)
    logger = get_log(name="demo", file_name="demo.log")

    if len(cfg.DEMO.HUMAN_PLAYERS) < 2:
        # init player
        model = get_model(game, cfg)
        cp = torch.load(cfg.DEMO.CHECKPOINT, map_location="cpu")
        model.load_state_dict(cp["state_dict"])
        model.eval()
        model.freeze_param()
        model.to(cfg.DEVICE)
        mcts = MCTS(game, cfg, model)
        AI_player = lambda x: np.argmax(mcts.get_policy_infer(x, temp=0))
    else:
        AI_player = None

    # init gui
    root = Tk()
    screen = Canvas(
        root,
        width=cfg.GUI.WIDTH,
        height=cfg.GUI.HEIGHT,
        background=COLOR[0],
        highlightthickness=0,
    )
    screen.pack()
    gui = GUI(root, screen, logger, COLOR, cfg)

    arena = ArenaGUI(AI_player, cfg.DEMO.HUMAN_PLAYERS, game, gui.update_state)

    # run
    gui.start(arena=arena)
    root.wm_title("Othello")
    root.mainloop()


if __name__ == "__main__":
    args = parse_args()
    if not args.config:
        args.config = "./configs/demo.yaml"
    cfg = get_cfg_defaults()
    cfg = setup_config(cfg, args)
    run_gui(cfg)
