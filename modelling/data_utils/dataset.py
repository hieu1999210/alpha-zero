import logging
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class EXP(Dataset):
    def __init__(self, data_folder, files, iter_list, augmentation=False):
        """
        args:
            iter_list: list of iteration index to load exp
        """
        super(EXP, self).__init__()

        states, policies, values = [], [], []

        for iter_id in iter_list:
            assert (
                f"iter_{iter_id:0>3}" in files[iter_id - 1]["data_name"]
            ), f"got iter {iter_id} and files_list: {files}"
            path = os.path.join(data_folder, files[iter_id - 1]["data_name"])
            data = np.load(path, allow_pickle=True).item()
            states.append(data["states"])
            policies.append(data["policies"])
            values.append(data["values"])

        self.states = np.concatenate(states, axis=0).astype(np.float32)
        self.policies = np.concatenate(policies, axis=0).astype(np.float32)
        self.values = np.concatenate(values, axis=0).astype(np.float32)

        logging.info(f"loaded {len(self)} samples")
        self.board_size = self.states.shape[-1]
        self.augmentation = augmentation

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        board = self.states[idx]
        policy = self.policies[idx]
        # print(f"origin board{idx}  \n {board}")
        if self.augmentation:
            rotate = np.random.choice([1, 2, 3, 4])
            is_flib = np.random.choice([True, False])
            policy_board = np.reshape(policy[:-1], (self.board_size, self.board_size))
            # print(f"origin policy{idx}  \n {policy_board}")
            # rotate
            board = np.rot90(board, rotate)
            policy_board = np.rot90(policy_board, rotate)

            # flip
            if is_flib:
                board = np.fliplr(board)
                policy_board = np.fliplr(policy_board)
            # print(f"aug board{idx}  \n {board}")
            # print(f"aug policy{idx}  \n {policy_board}")
            new_policy = np.zeros_like((policy))
            new_policy[:-1] = policy_board.reshape(-1)
            new_policy[-1] = policy[-1]
        else:
            new_policy = policy

        return (
            torch.tensor(board[np.newaxis, :, :].copy()),
            torch.tensor(new_policy.copy()),
            torch.tensor(self.values[idx]),
        )


class Batch:
    def __init__(self, data):
        """
        pad images to the same size
        """
        states, policies, values = list(zip(*data))

        self.states = torch.stack(states, 0)
        self.policies = torch.stack(policies, 0)
        self.values = torch.stack(values, 0)
        # print(self.policies[0])
        # print(self.states[0])

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
