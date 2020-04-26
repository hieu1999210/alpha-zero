from yacs.config import CfgNode as CN


_C = CN()

_C.MCTS = CN()
_C.MCTS.C_PUCT = 0.1
_C.MCTS.DIRICHLET_ALPHA = 0.03
_C.MCTS.DIRICHLET_WEIGHT = 0.25

_C.PRINT_FREQ = 50
_C.EXPERIMENT = "new_exp" # Experiment name
_C.DEBUG = False

# _C.INFER = CN()
# _C.INFER.TTA = False

_C.MODEL = CN()
_C.MODEL.NORM_LAYER = "bn2d"
# _C.MODEL.SEGMENT_UPSAMPLING = "nearest"
# _C.MODEL.SIGMOID = False

# _C.MODEL.ASPP = CN()
# _C.MODEL.ASPP.ATROUS_RATES = [1,2,3]
# _C.MODEL.ASPP.BOTTLE_NECK = 32
# _C.MODEL.ASPP.UP_SAMPLING = "trilinear"
# _C.MODEL.ASPP.DROP_OUT = 0.5

_C.MODEL.RETINANET = CN()
_C.MODEL.RETINANET.NUM_CONVS = 4
_C.MODEL.RETINANET.PRIOR_PROB = 0.01 # for bias in conv for cls head
_C.MODEL.RETINANET.IN_FEATURES = [f"p{i}" for i in range(3,8)]
# background and foreground thresholds repectively
_C.MODEL.RETINANET.IOU_THRESHOLDS = [0.4, 0.5] 
# 1: foreground, 0: background, -1: ignore
_C.MODEL.RETINANET.IOU_LABELS = [0, -1, 1]

# config for inference
_C.MODEL.RETINANET.NUM_TOPK = 1000
_C.MODEL.RETINANET.SCORE_THRESH = 0.05
_C.MODEL.RETINANET.MAX_BOXES_PER_IMAGE = 10
_C.MODEL.RETINANET.NMS_THRESH = 0.5

_C.MODEL.RPN = CN()
_C.MODEL.RPN.BB_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

_C.MODEL.FPN = CN()
_C.MODEL.FPN.OUT_CHANNELS = 256
_C.MODEL.FPN.UP_SAMPLING = "bilinear"
_C.MODEL.FPN.IN_FEATURES = ["layer2", "layer3", "layer4"]
_C.MODEL.FPN.TOP_BLOCK = True

_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = "resnet34"
_C.MODEL.BACKBONE.PRETRAINED = True

_C.MODEL.ANCHORS = CN()
_C.MODEL.ANCHORS.ASPECT_RATIOS = [[0.5, 1, 2]]*3 # [[0.5, 1, 2]]*5
_C.MODEL.ANCHORS.SIZES = [[x, x * 2**(1.0/3), x * 2**(2.0/3) ] for x in [32, 64, 128]]
# [[x, x * 2**(1.0/3), x * 2**(2.0/3) ] for x in [32, 64, 128, 256, 512]]
_C.MODEL.ANCHORS.OFFSET = 0.0 
# _C.MODEL.UNET = CN()
# _C.MODEL.UNET.NUM_LEVELS = 4

_C.SYSTEM = CN()
_C.SYSTEM.SEED = 0
# _C.SYSTEM.FP16 = True
_C.SYSTEM.OPT_L = "O0"
_C.SYSTEM.DEVICE = "cuda"
_C.SYSTEM.MULTI_GPU = False
_C.SYSTEM.NUM_WORKERS = 8

_C.DIRS = CN()
_C.DIRS.DATA = "/home/ad/LungCancer/dataset/split"
# _C.DIRS.WEIGHTS = "./weights/"
_C.DIRS.OUTPUTS = "/home/ad/hieu_mammo/lungct_segment"
# _C.DIRS.LOGS = "./logs/"
_C.DIRS.EXPERIMENT = ''

_C.DATA = CN()
# _C.DATA.AUGMENT_PROB = 0.5
# _C.DATA.MIXUP_PROB = 0.0
# _C.DATA.CUTMIX_PROB = 0.0
_C.DATA.INP_CHANNELS = 1
_C.DATA.NUM_CLASSES = 13
# _C.DATA.WINDOW_CENTER = 700
# _C.DATA.WINDOW_WIDTH = 2100
_C.DATA.BASE_WIDTH = 1000
_C.DATA.CROP_CSV = ""

_C.OPT = CN()
_C.OPT.OPTIMIZER = "adamw"
_C.OPT.GD_STEPS = 1 
_C.OPT.WARMUP_EPOCHS = 1
_C.OPT.BASE_LR = 1e-2
_C.OPT.WEIGHT_DECAY = 1e-2
_C.OPT.WEIGHT_DECAY_BIAS = 0.0
_C.OPT.MOMENTUM = 0.9

_C.LOSS = CN()
_C.LOSS.CLS_NAME = "sigmoid_focal_loss"
_C.LOSS.REG_NAME = "smooth_l1_loss"
_C.LOSS.REDUCTION = "sum"
_C.LOSS.SMOOTHL1_BETA = 0.1
_C.LOSS.REG_WEIGHT = 1.0
_C.LOSS.CLS_WEIGHT = 1.0
# focal loss
_C.LOSS.FOCAL = CN()
_C.LOSS.FOCAL.GAMMA = 2
_C.LOSS.FOCAL.ALPHA = 0.25



_C.TRAIN = CN()
_C.TRAIN.LABELS_JSON = "/home/ad/LungCancer/dataset/split/cubes_csv/train_fold0.csv"
_C.TRAIN.FOLD = 0
_C.TRAIN.EPOCHS = 20
_C.TRAIN.BATCH_SIZE = 4

_C.TRAIN.DROPOUT = 0.0


_C.VAL = CN()
_C.VAL.LABELS_JSON = "/home/ad/LungCancer/dataset/split/cubes_csv/val_fold0.csv"
_C.VAL.BATCH_SIZE = 1
# _C.VAL.MAIN_METRIC = 'dice_score'
# _C.VAL.SEG_THRESH = 0.5

_C.CONST = CN()
_C.CONST.LABELS = []

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`