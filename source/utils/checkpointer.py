import torch
import os
import json


class Checkpointer:
    """
    manage checkpoint
    
    working scenarios:
    i. training from scratch: (also include training from public pretraned wieght)
        pretrained weight (locally) and checkpoint are None
        
    ii. finetune: load pretrained weights (local) to model, 
        other checkpointables object are initialized from cfg
        pretrained_w is not None, and there is no existing checkpoint logs
        
    iii. resume: resume training procedure from existing checkpoint log,
        pretrained weight are

    """
    def __init__(self, cfg, logger, model, **checkpointables):
        """
        args: 
            
        """
        self.model = model
        self.checkpointables = checkpointables
        self.logger = logger
        self.save_freq = cfg.SOLVER.SAVE_CHECKPOINT_FREQ
        self.num_epochs = cfg.SOLVER.NUM_EPOCHS
        
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
        self.additional_info = {"epoch": 1, "iter": 1}
        
    def save_checkpoint(self, cp_name, **additional_info):
        """
        ensure that keywords of addtional info are different from checkpointables
        
        args:
            -- cp_name (str): checkpoint's file name
            -- additional_info: must have epoch
        """

        # update other info
        self.additional_info.update(additional_info)
        
        # save checkpoint
        current_epoch = additional_info["epoch"]
        is_last = current_epoch == self.num_epochs
        if ((current_epoch-1) % self.save_freq) == 0 or is_last:
            
            cp = {"state_dict": self.model.state_dict()}
            for key, ob in self.checkpointables.items():
                cp[key] = ob.state_dict()
            cp.update(self.additional_info)
            
            path = os.path.join(self.cp_folder, cp_name)
            torch.save(cp, path)        
            self._update_log(cp_name=cp_name)
            self.logger.info(f"saved new checkpoint to {path}")
    
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
        load pretrained weights (if any) or resume training
        args: 
            -- pretrained_w (str): path to pretrained weights
        return current epoch, current_best_metric
         
        NOTE: epoch numbering in logging and checkpoint start from 1,
            assume larger metric is better
        """
        # print("##########", pretrained_w)
        # scenario i.
        last_checkpoint = self.cp_logs["last_checkpoint"]
        if  last_checkpoint is not None:
            cp_path = os.path.join(self.cp_folder,last_checkpoint)
            state_dict = self._load_cp_file(cp_path)
            state_dict["epoch"] += 1
            self._load_state(state_dict)
            self.logger.info(f"resume from checkpoint {last_checkpoint}")
            return state_dict["epoch"]
        
        # scenario ii.
        elif pretrained_w is not None:
            weights = self._load_cp_file(pretrained_w)["state_dict"]
            self.model.load_state_dict(weights)
            self.cp_logs["pretrained"] = pretrained_w
            self.logger.info(f"finetune from pretrained weight: {pretrained_w}")
            return 1
        
        # scenario iii.
        else:
            self.logger.info("##### training from scratch #####")
            return 1

    def _update_log(self, cp_name=None, data_name=None):
        """
        update and save log
        """
        # update iter and epoch
        self.logs.update(self.additional_info)
        
        if cp_name is not None:
            self.logs["checkpoints"].append(cp_name)
            self.logs["last_checkpoint"] = cp_name
            
        if data_name is not None:
            self.logs["train_data"].append(data_name)

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