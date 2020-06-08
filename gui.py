#Library import
from tkinter import *
from math import *
from time import *
from random import *
from copy import deepcopy
from collections import deque
import numpy as np
import os
from MCTS import MCTS
from othello.OthelloGame import OthelloGame
from othello.pytorch.NNet import NNetWrapper as NNet
from ArenaGUI import ArenaGUI
from utils import dotdict
import logging
COLOR = {
    "valid": "#008000",
    -1: "#fff", # color for white tile
    1: "#000", # color for black tile
    0: "orange", # background color
    "text": "white",
    "undo": "#000088",
    "arrow": "white",
}

def get_log(name="main", folder=".", rank=0, file_name='logs.log', console=True):
    
    assert os.path.isdir(folder), f'log dir \'{folder}\' does not exist.'
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if rank > 0:
        return logger
    
    log_format = logging.Formatter(
        '{asctime}:{name}:  {message}',
        style='{'
    )
    if folder:
        fh = logging.FileHandler(os.path.join(folder, file_name))
        fh.setFormatter(log_format)
        logger.addHandler(fh)
    
    if console:
        ch = logging.StreamHandler(sys.stderr)
        ch.setFormatter(log_format)
        logger.addHandler(ch)
        
    return logger

class CONFIG:
    board_size = 8
    cell_size = 100
    tile_size = 80
    hint_size = 40
    button_size = 50
    border = 5
    mess_size = 100
    text_size = 40
    animation_delay = 0.001
    
CONFIG.width = (2 * (CONFIG.border + CONFIG.button_size) + 
                CONFIG.board_size * CONFIG.cell_size)

CONFIG.height = (CONFIG.board_size * CONFIG.cell_size +
                2 * CONFIG.border +
                CONFIG.button_size +
                CONFIG.mess_size)

CONFIG.base_cell = (CONFIG.border + CONFIG.button_size + 
            (CONFIG.cell_size - CONFIG.tile_size) // 2)

CONFIG.base_hint = (CONFIG.border + CONFIG.button_size + 
            (CONFIG.cell_size - CONFIG.hint_size) // 2)

CONFIG.base_player1_score = (
    CONFIG.border,
    CONFIG.height - CONFIG.border - CONFIG.mess_size
)

CONFIG.base_player2_score = (
    CONFIG.width - CONFIG.border - 2*CONFIG.text_size - CONFIG.hint_size - 15,
    CONFIG.height - CONFIG.border - CONFIG.mess_size
)


class GUI:
    """
    
    atributes:
        --current_state: (current_board, valid_moves, end_game, player_id, action)
            --board: current state of the game,
            --valid_moves: valid move of player on the current board
            --end_game: game end status of the current board
            --player_id: player to move from the current board
            --action: the last action that led to the current board
        
        --histories: list of state of the game
        -- 
    """
    def __init__(self, root, screen, board_size, logger, undo_size=3):
        self.screen = screen
        self.board_size = board_size
        self.root = root
        self.arena = None
        self.logger = logger
        screen.bind("<Button-1>", self.handle_click)
        screen.bind("<Key>", self.handle_key)
        screen.focus_set()

        
    def start(self, arena=None):
        """
        reset screen
        reset arena,
        reset histories
        reset player
        reset current_state
        """
        screen = self.screen
        
        # reset_screen
        screen.delete(ALL)
        self._display_button()
        self._display_background()
        self.turn_count = 0
        # reset arena
        if arena is not None:
            assert self.arena is None
            self.arena = arena
        else:
            assert self.arena is not None
        self.arena.reset()
        # return ########
    
        # reset history
        self.histories = deque()
        
        # reset current_state
        self.current_state = (
            np.zeros((CONFIG.board_size, CONFIG.board_size)),
            None, None, None, None
        )
        # update gui to initial state
        self.update_state(
            self.arena.board,
            self.arena.valid_moves,
            self.arena.end_game,
            self.arena.player_id,
            self.arena.action,
        )
        
        # AI play the first move
        self.arena.update()

    def handle_click(self, event):
        """
        
        """
        x_mouse = event.x
        y_mouse = event.y
        
        # undo
        undo_region = self._button_region["undo"]
        if self._check_region(x_mouse, y_mouse, undo_region):
            print("undo")
            self.undo()
            return
    
        # quit
        quit_region = self._button_region["quit"]
        if self._check_region(x_mouse, y_mouse, quit_region):
            print("quit")
            self.root.destroy()
            return
        
        cell_size = CONFIG.cell_size
        base_cell = CONFIG.base_cell
        x = (x_mouse - base_cell) // cell_size
        y = (y_mouse - base_cell) // cell_size
        print("click move", x, y)
        action = x*self.board_size + y
        self.arena.update(action)

    def _check_region(self, x , y, region):
        """
        check whether mouse is in a rectangle
        args:
            x,y: mouse coordinate
            region: (x1,y1, x2,y2) topleft and bottom right corners
        """
        return (
            x >= region[0] and 
            x <= region[2] and
            y >= region[1] and 
            y <= region[3]
        )
    
    def handle_key(self, event):
        """
        reload : r
        quit: q
        """
        symbol = event.keysym
        if symbol.lower()=="r":
            print("restart")
            self.start()
        elif symbol.lower()=="q":
            print("quit")
            self.root.destroy()

    def update_state(
        self, new_board, valid_moves, end_game, player, action, undo=False
    ):
        """
        display new game state after player making a move:
            display new board
            display score
            display end_game message
            display valid_move for next move (if human player)
        update state attributes and and history
        
        args:
            --new_board: next state of the game,
            --valid_moves: valid move of player on the new board
            --end_game: game end status of the new board
            --player: player to move from the new board (-1 or 1)
            --action: the action that led to the new board
            --undo: whether an actual move of undo
        
        the arguments are consistent with arena state
        
        NOTE: temperarily ignore undo  function
        
        """
        
        # NOTE: display_new_board require player who took the action
        if undo:
            self._display_new_board(new_board, -player, None)
        else:
            self._display_new_board(new_board, -player, action)
        
        # display score
        self._display_score()
        
        # display end game message
        if end_game:
            self._display_endgame(end_game)
            
        elif player == -1:
            # display valid move suggestions for human player
            self._display_valid_moves(valid_moves)
        self.turn_count += 1
        # update state attributes and histories
        state = (new_board, valid_moves, end_game, player, action)
        self.current_state = state
        self.histories.append(state)
        self.logger.info(
            f"turn: {self.turn_count}"
            f"\n{state[0].transpose()}"
            f"\nend_game: {state[2]}"
            f"\nplayer: {state[3]}"
            f"\naction: {state[4]}")
    
    def undo(self):
        """
        if current turn is human, reverse to previous human turn
        NOTE: if there is no human move in histories (i.e. len(histories) == 2)
        """
        
        if self.current_state[3] == 1:
            print("cannot undo AI turn")
            return
        if len(self.histories) <= 2:
            print("no turn to undo")
            return
        assert self.histories[-3][3] == -1, "something wrong"
        
        prev_state = self.histories[-3]
        self.histories.pop()
        self.histories.pop()
        self.histories.pop()
        # reverse arena state
        self.arena.board = prev_state[0]
        self.arena.valid_moves = prev_state[1]
        self.arena.end_game = prev_state[2]
        self.arena.player_id = prev_state[3]
        self.arena.action = prev_state[4]
        # reverse gui
        self.update_state(*prev_state, undo=True)
        

    def _display_endgame(self, game_result):
        self.screen.delete("end game")
        winner = "AI" if game_result == 1 else "human"
        self.screen.create_text(
            CONFIG.width//2,
            CONFIG.height - CONFIG.border - CONFIG.mess_size//2,
            anchor="c",
            tags="end game",
            font=("Consolas",CONFIG.text_size), 
            text=f"The winner is {winner}!")
    
    def _display_valid_moves(self, valid_moves):
        """
        display hint of valid move
        """
        x0 = CONFIG.base_hint
        cell_size = CONFIG.cell_size
        y0 = x0
        
        for move, valid in enumerate(valid_moves):
            if valid:
                x, y = move//self.board_size, move%self.board_size
                self.screen.create_oval(
                    x0 + x*cell_size, 
                    y0 + y*cell_size, 
                    x0 + x*cell_size + CONFIG.hint_size, 
                    y0 + y*cell_size + CONFIG.hint_size, 
                    tags="highlight",
                    fill=COLOR["valid"], 
                    outline=COLOR["valid"]
                )

    def _display_new_board(self, new_board, player, action):
        """
        display new_state:
            display action on old board (if any)
            shrink, grow (animation for board change)
        args:
            --new_board: new board to be displayed
            --action(tuple(int)): (x,y) the action that resulted in new_board
            --player(int): the one that took the action (1 or -1)
        
        NOTE: 
            -- compare new_board and self.current_state to update
                assume current_state is alreadly displayed
            -- player is the one who has taken the action, 
                different from player stored in Arena
            -- in undo case, action is "undo" and new_board is previous board
            
        """
        assert action is None or isinstance(action, tuple), \
            f"invalid action, got {action}"
        
        screen = self.screen
        board_size = self.board_size
        current_board = self.current_state[0]
        cell_size = CONFIG.cell_size
        tile_size = CONFIG.tile_size
        # base offset (x0,y0)
        x0 = CONFIG.base_cell
        y0 = x0
        # print("x0", x0)
        screen.delete("highlight")
        
        # sentinel action
        if action is not None:
            _x, _y = action
        else:
            _x, _y = -1, -1
    
        # display action
        if action is not None:
            x, y = _x, _y
            screen.delete(f"tile {x}-{y}")
            screen.create_oval(
                x0 + x*CONFIG.cell_size, 
                y0 + y*CONFIG.cell_size, 
                x0 + x*CONFIG.cell_size + CONFIG.tile_size, 
                y0 + y*CONFIG.cell_size + CONFIG.tile_size, 
                tags=f"tile {x}-{y}",
                fill=COLOR[player], 
                outline=COLOR[player]
            )
            screen.update()
        
        # display new state
        half_tile = CONFIG.tile_size//2
        for x in range(board_size):
            for y in range(board_size):
                
                # skip animation on action tile
                if x == _x and y == _y:
                    continue
                
                # animation
                current_tile = current_board[x,y]
                new_tile = new_board[x,y]
                if current_tile != new_tile:
                    screen.delete(f"{x}-{y}")
                    # get cell coordinate
                    x1 = x0 + x*cell_size
                    x2 = x1 + tile_size
                    y1 = y0 + y*cell_size
                    y2 = y1 + tile_size
                    
                    ## shrink 
                    for i in range(half_tile):
                        screen.create_oval(
                            x1+i, y1+i, x2-i, y2-i,
                            tags="tile animated", 
                            fill=COLOR[current_tile], 
                            outline=COLOR[current_tile])
                        # if i%30 == 0:
                        sleep(CONFIG.animation_delay)
                        screen.update()
                        screen.delete("animated")
                    
                    ## grow
                    for i in reversed(range(half_tile)):
                        screen.create_oval(
                            x1+i, y1+i, x2-i, y2-i,
                            tags="tile animated",
                            fill=COLOR[new_tile], 
                            outline=COLOR[new_tile])
                        # if i%30 == 0:
                        sleep(CONFIG.animation_delay)
                        screen.update()
                        screen.delete("animated")
                    
                    screen.create_oval(
                        x1, y1, x2, y2, tags=f"tile {x}-{y}",
                        fill=COLOR[new_tile], outline=COLOR[new_tile])
                    screen.update()
    
    def _display_score(self):
        """
        calulate current score from current_state
        clear old score and write new score
        
        NOTE: get score by self.arena.get_score()
        """
        screen = self.screen
        screen.delete("score")
        score = self.arena.get_scores()
        
        # display player1's score
        base1 = CONFIG.base_player1_score
        x1 = base1[0]
        y1 = base1[1] + (CONFIG.mess_size - CONFIG.hint_size)//2
        screen.create_oval(
            x1, y1, x1 + CONFIG.hint_size, y1 + CONFIG.hint_size, 
            tags="score",
            fill=COLOR[1], 
            outline=COLOR[1]
        )
        screen.create_text(
            x1 + CONFIG.hint_size + 15,
            CONFIG.height - CONFIG.border - CONFIG.mess_size//2,
            anchor="w",
            tags="score",
            font=("Consolas",CONFIG.text_size), 
            text=str(score[1])
        )
        
        # display player1's score
        base1 = CONFIG.base_player2_score
        x1 = base1[0]
        y1 = base1[1] + (CONFIG.mess_size - CONFIG.hint_size)//2
        screen.create_oval(
            x1, y1, x1 + CONFIG.hint_size, y1 + CONFIG.hint_size, 
            tags="score",
            fill=COLOR[-1], 
            outline=COLOR[-1]
        )
        screen.create_text(
            x1 + CONFIG.hint_size + 15,
            CONFIG.height - CONFIG.border - CONFIG.mess_size//2,
            anchor="w",
            tags="score",
            font=("Consolas",CONFIG.text_size), 
            text=str(score[-1])
        )
    
    def _display_button(self):
        """
        display undo and quit button
        and store their regions (x1, y1, x2, y2)
        """
        screen = self.screen
        
        # undo button (top_left)
        ## background
        undo_region = (
            CONFIG.border, CONFIG.border, 
            CONFIG.border + CONFIG.button_size, 
            CONFIG.border + CONFIG.button_size
        )
        screen.create_rectangle(*undo_region, fill="#000033", outline="#000033")
        ##arrow
        screen.create_arc(10,10,50,50,fill="#000088", width="2",style="arc",outline="white",extent=300)
        screen.create_polygon(38,43,41,50,45,44,fill="white",outline="white")

        # quit buttom
        ## background
        quit_region = (
            CONFIG.width - CONFIG.border - CONFIG.button_size, 
            CONFIG.border, 
            CONFIG.width - CONFIG.border, 
            CONFIG.border + CONFIG.button_size
        )
        screen.create_rectangle(*quit_region, fill="#880000", outline="#880000")
        ##x
        screen.create_line(
            quit_region[0] + CONFIG.border, quit_region[1] + CONFIG.border,
            quit_region[2] - CONFIG.border, quit_region[3] - CONFIG.border,
            fill="white",width="3"
        )
        screen.create_line(
            quit_region[0] + CONFIG.border, quit_region[3] - CONFIG.border,
            quit_region[2] - CONFIG.border, quit_region[1] + CONFIG.border,
            fill="white",width="3"
        )
        self._button_region = {
            "undo": undo_region,
            "quit": quit_region,
        }
        
    def _display_background(self):
        screen = self.screen
        #Drawing the intermediate lines
        shift = CONFIG.border + CONFIG.button_size + CONFIG.cell_size
        line_range = (
            CONFIG.border + CONFIG.button_size,
            CONFIG.border + CONFIG.button_size + CONFIG.board_size * CONFIG.cell_size
        )
        for _ in range(CONFIG.board_size - 1):
            #Horizontal line
            screen.create_line(
                line_range[0], shift, line_range[1], shift, fill="#111")

            #Vertical line
            screen.create_line(
                shift, line_range[0], shift, line_range[1], fill="#111")
            
            shift += CONFIG.cell_size
        screen.update()

def main():
    root = Tk()
    screen = Canvas(
        root, width=CONFIG.width, height=CONFIG.height, 
        background=COLOR[0], highlightthickness=0
    )
    screen.pack()
    game = OthelloGame(CONFIG.board_size)
    logger = get_log()
    # init player 

    model = NNet(game)
    # n1.load_checkpoint('./pretrained_models/othello/pytorch/','6x100x25_best.pth.tar')
    model.load_checkpoint('./pretrained_models/othello/pytorch/','8x8_100checkpoints_best.pth.tar')
    args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
    mcts1 = MCTS(game, model, args1)
    AI_player = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

    gui = GUI(root, screen, CONFIG.board_size, logger)
    arena = ArenaGUI(AI_player, game, gui.update_state)
    
    gui.start(arena=arena)
    #Run forever
    root.wm_title("Othello - developed by nhom7")
    root.mainloop()

if __name__ == "__main__":
    main()