from .othello import OthelloGame, build_othello_game

def build_game(cfg):
    if cfg.GAME.NAME == "othello":
        return build_othello_game(cfg.GAME.BOARD_SIZE)
    else:
        raise ValueError("invalid game name")

