import torch
import os
import json


class Checkpointer:
    """
    manage checkpoint
    
    working scenarios:
    i. training from scratch: train_data is empty
        
    ii. start at selfplaying: last cp's iter == last train_data's iter
        
    iii. start at training: last cp's iter < last train_data's iter

    """
    def __init__(self, cfg, logger, model, **checkpointables):
        
        """
        NOTE: currently only support save last checkpoint only
        args: 
            
        """
        # assert cfg.SOLVER.SAVE_LAST_ONLY,\
            # "not support save serveral cp in an iteration"
        self.model = model
        self.checkpointables = checkpointables
        self.logger = logger
        self.save_freq = cfg.SOLVER.SAVE_CHECKPOINT_FREQ
        self.num_epochs = cfg.SOLVER.NUM_EPOCHS
        if cfg.SOLVER.SAVE_LAST_ONLY:
            self.save_freq = self.num_epochs + 1
        # make checkpoint dirs
        cp_folder = os.path.join(cfg.DIRS.EXPERIMENT, "checkpoints")
        if not os.path.exists(cp_folder): 
            os.makedirs(cp_folder)
        self.cp_folder = cp_folder
        
        # make data dirs
        data_folder = os.path.join(cfg.DIRS.EXPERIMENT, "training_data")
        if not os.path.exists(data_folder): 
            os.makedirs(data_folder)
        self.data_folder = data_folder
        
        self.log_path = os.path.join(cfg.DIRS.EXPERIMENT, "cp_logs.json")
        
        # init logs
        self.logs = self._load_log() # if there is no existing logs, get dict with None values
        
        # store addtional info for each checkpoints
        self.additional_info = {"epoch": 0, "iter": 0}
        
        # latest model cp to update best model later
        self.latest_model = None
    
    def save_checkpoint(self, cp_name, is_best=False, **additional_info):
        """
        ensure that keywords of addtional info are different from checkpointables
        
        args:
            -- cp_name (str): checkpoint's file name
            -- additional_info: must have epoch
        """

        # update other info
        self.additional_info.update(additional_info)
        lastest_info = {"cp_name": cp_name}
        lastest_info.update(additional_info)
        self.latest_model = lastest_info
        
        # save checkpoint
        current_epoch = additional_info["epoch"]
        is_last = current_epoch == self.num_epochs
        if ((current_epoch) % self.save_freq) == 0 or is_last:
            
            cp = {"state_dict": self.model.state_dict()}
            for key, ob in self.checkpointables.items():
                cp[key] = ob.state_dict()
            cp.update(self.additional_info)
            
            path = os.path.join(self.cp_folder, cp_name)
            torch.save(cp, path)        
            self._update_log(cp_name=cp_name)
            self.logger.info(f"saved new checkpoint to {path}")
            
            # if is_last and additional_info["iter"] == 1:
            #     self.update_best_cp()
    
    def save_data(self, file_name, data, **additional_info):
        """
        save experience data
        args:
            file_name (str): data file_name
            data (dict): exp data
            additional_info (dict): must have iter 
        NOTE: current epoch is reseted to zero 
        """
        
        # update other info
        self.additional_info.update(additional_info)
        self.additional_info["epoch"] = 0
        
        path = os.path.join(self.data_folder, file_name)
        torch.save(data, path)
        self._update_log(data_name=file_name)
        self.logger.info(f"saved new data to {path}")
        
    def load_resume(self, pretrained_w=None):
        """
        working scenarios:
        i. training from scratch: train_data is empty
            
        ii. last cp's iter == last train_data's iter
            
        iii. start at training: last cp's iter < last train_data's iter
    
        load pretrained weights (if any) or resume training
        
        args: 
            -- pretrained_w (str): path to pretrained weights
        return current epoch, current iter
        epoch 0 means start selfplaying
        epoch >0: skip selfplaying, start train model
        iter: current iter
        
        NOTE: epoch numbering in logging and checkpoint start from 1,
            assume larger metric is better
        """
        assert pretrained_w is None, \
            "currently do not support pretrained_weights"
        logs = self.logs
        last_cp_iter = 0
        if len(logs["checkpoints"]) > 0:
            last_cp_iter = logs["checkpoints"][-1]["iter"]
        
        last_data_iter = 0
        if len(logs["train_data"]) > 0:
            last_data_iter = logs["train_data"][-1]["iter"]
        
        assert (last_cp_iter <= last_data_iter and 
            last_cp_iter >= last_data_iter-1), \
            "something wrong in cp logs"
        
        # scenario i.
        if  last_data_iter == 0:
            self.logger.info("##### training from scratch #####")
            return 0, 1
        
        # scenario ii. last_data_iter = last_data_iter > 0
        elif last_data_iter == last_cp_iter:
            last_checkpoint = logs["checkpoints"][-1]["cp_name"]
            cp_path = os.path.join(self.cp_folder, last_checkpoint)
            state_dict = torch.load(cp_path)
            self._load_state(state_dict)
            self.logger.info(f"resume from checkpoint {last_checkpoint}")

            if state_dict["epoch"] == self.num_epochs:
                self.logger.info(f"starting from iter {last_data_iter + 1}")
                return 0, last_data_iter + 1
            else:
                self.logger.info(f"starting from iter {last_data_iter}")
                return state_dict["epoch"] + 1, last_data_iter
        
        # last_cp_iter < last_data_iter 
        # finish selfplaying but havent train 
        else: 
            if last_cp_iter == 0:
                self.logger.info(
                    "##### training from scratch, got train_data iter0 #####")
                return 1, 1
            else:
                last_checkpoint = logs["checkpoints"][-1]["cp_name"]
                cp_path = os.path.join(self.cp_folder, last_checkpoint)
                state_dict = torch.load(cp_path)
                # assert state_dict["epoch"] == self.num_epochs
                self._load_state(state_dict)
                self.logger.info(f"resume from checkpoint {last_checkpoint}")
                
                self.logger.info(f"starting from iter {last_data_iter}")
                return 1, last_data_iter

    def get_best_cp_name(self):
        if self.logs["best_cp"] is not None:
            return self.logs["best_cp"]["cp_name"]
        return None
    
    def load_best_cp(self, model):
        """
        load checkpoint of previous iter for comparision
        """
        # assert self.additional_info["iter"] > 0
        if self.logs["best_cp"] is not None:
            cp_name = self.logs["best_cp"]["cp_name"]
            path = os.path.join(self.cp_folder, cp_name)
            cp = torch.load(path, map_location="cuda")
            model.load_state_dict(cp["state_dict"])
            return
        print("no best cp")
    
    def get_current_cp_name(self):
        if self.latest_model is not None:
            return self.latest_model["cp_name"]
        return None
    
    def update_best_cp(self):
        self.logs["best_cp"] = self.latest_model

    def _update_log(self, cp_name=None, data_name=None):
        """
        update and save log
        """
        # update iter and epoch
        self.logs.update(self.additional_info)
        
        if cp_name is not None:
            cp_info = {
                "cp_name": cp_name, 
                "iter": self.additional_info["iter"],
                "epoch": self.additional_info["epoch"]
            }
            self.logs["checkpoints"].append(cp_info)
            self.logs["last_checkpoint"] = cp_info
            
        if data_name is not None:
            data_info = {
                "data_name": data_name, 
                "iter": self.additional_info["iter"],
                "epoch": self.additional_info["epoch"]
            }
            self.logs["train_data"].append(data_info)

        # save new logs 
        with open(self.log_path, "w") as f:
            json.dump(self.logs, f, indent=4)
    
    def _load_log(self):
        """
        only load at init
        """

        if os.path.exists(self.log_path):
            with open(self.log_path, "r") as f:
                return json.load(f)
            
        self.logger.info("There is no existing checkpoint log")
        return {
            "pretrained": None,
            "last_checkpoint": None,
            "epoch": 0,
            "iter": 0,
            "train_data": [],
            "checkpoints": [],
            "best_cp": None,
        }
    
    def _load_state(self, state_dict):
        """
        load state to model and checkpointable objects and other info
        """
        self.model.load_state_dict(state_dict.pop("state_dict"))
        for key, ob in self.checkpointables.items():
            ob.load_state_dict(state_dict.pop(key))
            
        for key, info in state_dict.items():
            self.additional_info[key] = info