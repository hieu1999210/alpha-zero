from .data_utils import get_dataset
from .model import OthelloNet, Resnet
from .model_trainer import ModelTrainer


def get_model(game, cfg):
    name = cfg.MODEL.NAME
    if name == "OthelloNet":
        return OthelloNet(game)
    elif name == "Resnet":
        return Resnet(game, cfg)
