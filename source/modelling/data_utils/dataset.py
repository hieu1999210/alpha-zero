

import os
# import json
import logging

import torch
import numpy as np
from torch.utils.data import Dataset


class EXP(Dataset):
    def __init__(self, data_folder, files, iter_list):
        """
        args:
            iter_list: list of iteration index to load exp
        """
        super(EXP, self).__init__()
        
        states, policies, values = [], [], []

        for iter_id in iter_list:
            assert f"iter_{iter_id:0>3}" in files[iter_id-1]["data_name"], \
                f"got iter {iter_id} and files_list: {files}"
            path = os.path.join(data_folder, files[iter_id-1]["data_name"])
            data = torch.load(path)
            states.append(data["states"])
            policies.append(data["policies"])
            values.append(data["values"])
        
        self.states = torch.cat(states, dim=0).type(torch.float)
        self.policies = torch.cat(policies, dim=0).type(torch.float)
        self.values = torch.cat(values, dim=0).type(torch.float)

        logging.info(f"loaded {len(self)} samples")
    
    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx]


class Batch:
    
    def __init__(self, data):
        """
        pad images to the same size
        """
        states, policies, values = list(zip(*data))
        
        self.states = torch.stack(states, 0)
        self.policies = torch.stack(policies, 0)
        self.values = torch.stack(values, 0)

    def __len__(self):
        return len(self.states)

    def cuda(self):
        self.states = self.states.cuda()
        self.policies = self.policies.cuda()
        self.values = self.values.cuda()
        
    def pin_memmory(self):
        self.states = self.states.pin_memory()
        self.policies = self.policies.pin_memory()
        self.values = self.values.pin_memory()