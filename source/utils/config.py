from yacs.config import CfgNode as CN


_C = CN()

_C.DEVICE = "cuda"
_C.MCTS = CN()
_C.MCTS.C_PUCT = 0.1
_C.MCTS.DIRICHLET_ALPHA = 0.03
_C.MCTS.DIRICHLET_WEIGHT = 0.25
_C.MCTS.NUM_SIMULATION_PER_STEP = 64

_C.SELF_PLAY = CN()
_C.SELF_PLAY.GAME_PER_ITER = 128
_C.SELF_PLAY.BATCH_SIZE = 4
_C.SELF_PLAY.NUM_WORKER = 4
_C.SELF_PLAY.TEMP_THRESH = 30
_C.SELF_PLAY.VERBOSE_FREQ = 16
# match config to compare model
_C.MATCH = CN()
_C.MATCH.NUM_MATCHES = 64
_C.MATCH.NUM_WORKERS = 4
_C.MATCH.VERBOSE_FREQ = 4
# level of exploration in a match
_C.MATCH.TEMP = 0

_C.GAME = CN()
_C.GAME.NAME = "othello"
_C.GAME.DEFAULT_PLAYER = 1
_C.GAME.FIRST_PLAYER = 1
_C.GAME.BOARD_SIZE = 8

_C.DIRS = CN()
_C.DIRS.OUTPUTS = "/mnt/DATA/learning_stuffs/uni/20192/artificial intelligence/project/alpha-zero/runs/"
_C.DIRS.EXPERIMENT = '.'

_C.MODEL = CN()
_C.MODEL.NORM_LAYER = "bn2d"
_C.MODEL.BASE_CHANNELS = 128
_C.MODEL.BOTTLENECK_CHANNELS = 32
_C.MODEL.NUM_BLOCKS = 19

_C.SOLVER = CN()
_C.SOLVER.NUM_ITERS = 3
_C.SOLVER.NUM_EPOCHS = 10
_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.GD_STEPS = 1
_C.SOLVER.WARMUP_EPOCHS = 1
_C.SOLVER.WEIGHT_DECAY = 0.01
_C.SOLVER.BATCH_SIZE = 32
_C.SOLVER.SAVE_CHECKPOINT_FREQ = 1
_C.SOLVER.SAVE_LAST_ONLY = False
_C.SOLVER.UPDATE_THRESH = 0.6
_C.EXPERIMENT = ""

_C.DATA = CN()
_C.DATA.NUM_WORKERS = 8
_C.DATA.HISTORY_LENGTH = 4


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`