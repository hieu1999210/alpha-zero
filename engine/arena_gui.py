import logging

log = logging.getLogger(__name__)


class ArenaGUI:
    """
    An Arena class where any 2 agents can be pit against each other.
    black : player_id = 1, player1
    white : player_id = -1, player2

    game state attributes:
        --board: current state of the game,
        --valid_moves: valid move of player on the current board
        --end_game: game end status of the current board
        --player_id: player to move from the new board
        --action: the last action that led to the current board
    """

    def __init__(self, AI_player, human_player_ids, game, display):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.
            assume player 2 is human

        args:
            --player1: funtion to return action, given a state (AI)
            --game(Game)
            --display: a method to update gui given new state

        """
        self.AI_player = AI_player
        self.human_players = human_player_ids
        self.game = game
        self.display = display
        self.board_size = game.getBoardSize()

    def reset(self):
        """
        reset game state to inital state
        """
        self.player_id = 1
        self.board = self.game.getInitBoard()
        self.valid_moves = self.game.getValidMoves(self.board, self.player_id)
        self.end_game = 0
        self.action = None

    def update(self, action=None):
        """
        make move by taking action and update states attributes
        run recursively until human turn (if has valid move) or end game

        if human player:
            if has valid action, but given invalid action: do nothing
            if has no valid action: next turn
            if given valid action: act and update state

        if computer player:
            act and update

        args:
            -- action(int): x*board_size + y
        """
        board = self.board
        player_id = self.player_id
        valid_moves = self.valid_moves
        board_size = self.board_size

        # human player
        if player_id in self.human_players:
            # no valid action, change action to do nothing
            if valid_moves[-1] == 1:
                assert action is None
                action = len(valid_moves) - 1

            # the given action is invalid, skip this step
            elif action is not None and valid_moves[action] == 0:
                print("invalid move")
                return

        # AI player
        else:
            action = self.AI_player(self.game.getCanonicalForm(board, player_id))

        if valid_moves[action] == 0:
            log.error(f"Action {action} is not valid!")
            log.debug(f"valids = {valid_moves}")
            raise ValueError("invalid move")

        # get new state
        board, player_id = self.game.getNextState(board, player_id, action)
        valid_moves = self.game.getValidMoves(board, player_id)
        end_game = self.game.getGameEnded(board, player_id) * player_id

        # convert action for caching
        if action == len(valid_moves) - 1:
            action = None
        else:
            board_size = self.game.getBoardSize()[0]
            x, y = action // board_size, action % board_size
            action = (x, y)

        # update game state
        self.board = board
        self.valid_moves = valid_moves
        self.end_game = end_game
        self.player_id = player_id
        self.action = action

        # update gui
        self.display(
            new_board=board,
            valid_moves=valid_moves,
            end_game=end_game,
            player=player_id,
            action=action,
        )

        # recusive condition
        if self.end_game == 0:

            # next turn is AI's
            if self.player_id not in self.human_players:
                self.update()

            # next turn is human but no valid move
            elif self.valid_moves[-1] == 1:
                print("no valid move for human")
                self.update()

    def get_scores(self):
        player1 = (self.board == 1).sum()
        player2 = (self.board == -1).sum()
        return {1: player1, -1: player2}
