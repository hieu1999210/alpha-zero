from .model import Model, OthelloNNet
from .data_utils import get_dataset
from .model_trainer import ModelTrainer

def get_model(game, cfg):
    # return Model(game, cfg)
    return OthelloNNet(game)