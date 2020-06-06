from ArenaGUI import ArenaGUI
from gui import *
from MCTS import MCTS
from othello.OthelloGame import OthelloGame
from othello.OthelloPlayers import *
from othello.pytorch.NNet import NNetWrapper as NNet

from tkinter import *
import numpy as np
from utils import *

mini_othello = False  # Play in 6x6 instead of the normal 8x8.
human_vs_cpu = True

if mini_othello:
    g = OthelloGame(6)
else:
    g = OthelloGame(8)


# init player
## human player
hp = HumanOthelloPlayer(g).play

## AI player
n1 = NNet(g)
if mini_othello:
    n1.load_checkpoint('./pretrained_models/othello/pytorch/','6x100x25_best.pth.tar')
else:
    n1.load_checkpoint('./pretrained_models/othello/pytorch/','8x8_100checkpoints_best.pth.tar')
args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

if human_vs_cpu:
    player2 = hp
else:
    n2 = NNet(g)
    n2.load_checkpoint('./pretrained_models/othello/pytorch/', '8x8_100checkpoints_best.pth.tar')
    args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    mcts2 = MCTS(g, n2, args2)
    n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

    player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.
    
# init GUI
root = Tk()
screen = Canvas(root, width=500, height=600, background="#222",highlightthickness=0)
screen.pack()

init_game(screen)

gui = GUI(screen)
# init arena
arena = ArenaGUI(n1p, player2, g, display=OthelloGame.display)
# run GUI by click

arena.playGame()