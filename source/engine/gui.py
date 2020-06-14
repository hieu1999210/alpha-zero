from tkinter import ALL

from time import sleep

from collections import deque
import numpy as np
import os

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
    
    def __init__(self, root, screen, logger, color, cfg):
        self.screen         = screen
        self.root           = root
        self.arena          = None
        self.logger         = logger
        self.cfg            = cfg
        self.color          = color
        self.human_players  = cfg.DEMO.HUMAN_PLAYERS
        
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
        board_size = self.cfg.GAME.BOARD_SIZE
        
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
            np.zeros((board_size, board_size)),
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
        
        # AI is the first player
        if self.cfg.GAME.FIRST_PLAYER not in self.human_players:
            self.arena.update()

    def handle_click(self, event):
        """
        
        """
        x_mouse = event.x
        y_mouse = event.y
        board_size = self.cfg.GAME.BOARD_SIZE
        
        # undo
        undo_region = self._button_region["undo"]
        if self._check_region(x_mouse, y_mouse, undo_region):
            print("undo")
            self.undo()
            return
    
        # quit
        # quit_region = self._button_region["quit"]
        # if self._check_region(x_mouse, y_mouse, quit_region):
        #     print("quit")
        #     self.root.destroy()
        #     return
        
        # move
        cell_size = self.cfg.GUI.CELL_SIZE
        base_cell = self.cfg.GUI.BASE_CELL
        x = (x_mouse - base_cell) // cell_size
        y = (y_mouse - base_cell) // cell_size
        print("click move", x, y)
        action = x*board_size + y
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
            
        elif player in self.human_players:
            # display valid move suggestions for human player
            self._display_valid_moves(valid_moves)
        self.turn_count += 1
        # update state attributes and histories
        state = (new_board, valid_moves, end_game, player, action)
        self.current_state = state
        self.histories.append(state)
        if self.cfg.DEMO.VERBOSE:
            self.logger.info(
                f"turn: {self.turn_count}"
                f"\n{state[0].transpose()}"
                f"\nend_game: {state[2]}"
                f"\nplayer: {state[3]}"
                f"\naction: {state[4]}")
        
    def undo(self):
        """
        if current turn is human, reverse to previous human turn

        """
        self.screen.delete("text")
        
        # current is AI turn
        if self.current_state[3] not in self.human_players:
            print("cannot undo AI turn")
            return
        
        # no turn to undo
        if len(self.histories) == 1:
            print("no turn to undo")
            return
        if len(self.histories) == 2 and self.cfg.GAME.FIRST_PLAYER not in self.human_players:
            print("no turn to undo")
            return

        
        # reverse history to a human turn
        prev_state = [None]*5
        self.histories.pop()
        while prev_state[3] not in self.human_players:
            prev_state = self.histories.pop()
        # print("hist leng##########", len(self.histories))
        # print("prev ###########", prev_state)
        # reverse arena state
        self.arena.board = prev_state[0]
        self.arena.valid_moves = prev_state[1]
        self.arena.end_game = prev_state[2]
        self.arena.player_id = prev_state[3]
        self.arena.action = prev_state[4]
        # reverse gui
        self.update_state(*prev_state, undo=True)
        
    def _display_endgame(self, game_result):
        self.screen.delete("text")
        winner = "BLACK" if game_result == 1 else "WHITE"
        self.screen.create_text(
            self.cfg.GUI.WIDTH//2,
            self.cfg.GUI.HEIGHT - self.cfg.GUI.BORDER - self.cfg.GUI.MESS_SIZE//2,
            anchor="c",
            tags="text",
            font=("Consolas",self.cfg.GUI.TEXT_SIZE), 
            text=f"{winner} wins.")
    
    def _display_valid_moves(self, valid_moves):
        """
        display hint of valid move
        """
        x0          = self.cfg.GUI.BASE_HINT
        cell_size   = self.cfg.GUI.CELL_SIZE
        hint_size   = self.cfg.GUI.HINT_SIZE
        color       = self.color
        board_size  = self.cfg.GAME.BOARD_SIZE
        y0 = x0
        
        for move, valid in enumerate(valid_moves):
            if valid:
                x, y = move//board_size, move%board_size
                self.screen.create_oval(
                    x0 + x*cell_size, 
                    y0 + y*cell_size, 
                    x0 + x*cell_size + hint_size, 
                    y0 + y*cell_size + hint_size, 
                    tags="highlight",
                    fill=color["valid"], 
                    outline=color["valid"]
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
        
        screen          = self.screen
        board_size      = self.cfg.GAME.BOARD_SIZE
        current_board   = self.current_state[0]
        cell_size       = self.cfg.GUI.CELL_SIZE
        tile_size       = self.cfg.GUI.TILE_SIZE
        color           = self.color
        
        # base offset (x0,y0)
        x0 = self.cfg.GUI.BASE_CELL
        y0 = x0
        # print("x0", x0)
        screen.delete("highlight")
        screen.delete("text")
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
                x0 + x*cell_size, 
                y0 + y*cell_size, 
                x0 + x*cell_size + tile_size, 
                y0 + y*cell_size + tile_size, 
                tags=f"tile {x}-{y}",
                fill=color[player], 
                outline=color[player]
            )
            screen.update()
        
        # display new state
        half_tile = tile_size//2
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
                            fill=color[current_tile], 
                            outline=color[current_tile])
                        # if i%30 == 0:
                        sleep(self.cfg.GUI.ANIMATION_DELAY)
                        screen.update()
                        screen.delete("animated")
                    
                    ## grow
                    for i in reversed(range(half_tile)):
                        screen.create_oval(
                            x1+i, y1+i, x2-i, y2-i,
                            tags="tile animated",
                            fill=color[new_tile], 
                            outline=color[new_tile])
                        # if i%30 == 0:
                        sleep(self.cfg.GUI.ANIMATION_DELAY)
                        screen.update()
                        screen.delete("animated")
                    
                    screen.create_oval(
                        x1, y1, x2, y2, tags=f"tile {x}-{y}",
                        fill=color[new_tile], outline=color[new_tile])
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
        hint_size = self.cfg.GUI.HINT_SIZE
        mess_size = self.cfg.GUI.MESS_SIZE
        color = self.color
        
        # display player1's score
        base1 = self.cfg.GUI.BASE_PLAYER1_SCORE
        x1 = base1[0]
        y1 = base1[1] + (mess_size - hint_size)//2
        screen.create_oval(
            x1, y1, x1 + hint_size, y1 + hint_size, 
            tags="score",
            fill=color[1], 
            outline=color[1]
        )
        screen.create_text(
            x1 + hint_size + 15,
            self.cfg.GUI.HEIGHT - self.cfg.GUI.BORDER - mess_size//2,
            anchor="w",
            tags="score",
            font=("Consolas",self.cfg.GUI.TEXT_SIZE), 
            text=str(score[1])
        )
        
        # display player1's score
        base1 = self.cfg.GUI.BASE_PLAYER2_SCORE
        x1 = base1[0]
        y1 = base1[1] + (mess_size - hint_size)//2
        screen.create_oval(
            x1, y1, x1 + hint_size, y1 + hint_size, 
            tags="score",
            fill=color[-1], 
            outline=color[-1]
        )
        screen.create_text(
            x1 + hint_size + 15,
            self.cfg.GUI.HEIGHT - self.cfg.GUI.BORDER - mess_size//2,
            anchor="w",
            tags="score",
            font=("Consolas",self.cfg.GUI.TEXT_SIZE), 
            text=str(score[-1])
        )
    
    def _display_button(self):
        """
        display undo and quit button
        and store their regions (x1, y1, x2, y2)
        """
        screen = self.screen
        border = self.cfg.GUI.BORDER
        button_size = self.cfg.GUI.BUTTON_SIZE
        
        # undo button (top_left)
        ## background
        undo_region = (
            border, border, 
            border + button_size, border + button_size
        )
        screen.create_rectangle(*undo_region, fill="#000033", outline="#000033")
        ##arrow
        screen.create_arc(
            10,10,50,50,
            fill="#000088", width="2",style="arc",
            outline="white",extent=300)
        screen.create_polygon(38,43,41,50,45,44,fill="white",outline="white")

        # quit buttom
        # ## background
        # quit_region = (
        #     self.cfg.GUI.WIDTH - border - button_size, border, 
        #     self.cfg.GUI.WIDTH - border, border + button_size
        # )
        # screen.create_rectangle(*quit_region, fill="#880000", outline="#880000")
        # ##x
        # screen.create_line(
        #     quit_region[0] + border, quit_region[1] + border,
        #     quit_region[2] - border, quit_region[3] - border,
        #     fill="white",width="3"
        # )
        # screen.create_line(
        #     quit_region[0] + border, quit_region[3] - border,
        #     quit_region[2] - border, quit_region[1] + border,
        #     fill="white",width="3"
        # )
        self._button_region = {
            "undo": undo_region,
            # "quit": quit_region,
        }
        
    def _display_background(self):
        screen = self.screen
        cell_size = self.cfg.GUI.CELL_SIZE
        border = self.cfg.GUI.BORDER
        button_size = self.cfg.GUI.BUTTON_SIZE
        board_size = self.cfg.GAME.BOARD_SIZE
        #Drawing the intermediate lines
        shift = border + button_size + cell_size
        line_range = (
            border + button_size,
            border + button_size + board_size * cell_size
        )
        for _ in range(board_size - 1):
            #Horizontal line
            screen.create_line(
                line_range[0], shift, line_range[1], shift, fill="#111")

            #Vertical line
            screen.create_line(
                shift, line_range[0], shift, line_range[1], fill="#111")
            
            shift += cell_size
        screen.update()

