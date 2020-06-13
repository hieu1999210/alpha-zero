import torch
import numpy as np
import random

def set_deterministic(seed=0):
    """ 
    set seed for all build-in random generators
    and disable un-deterministic in torch
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)