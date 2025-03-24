import logging
import random
import math
import copy
import torch
import numpy as np
import torch.nn.functional as F
from core.clients.clienttarget_base import Client

class Clientaffcl(Client):
    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, incremental_train_data,incremental_test_data,args, device,
                 model_trainer):
        super().__init__(self, client_idx, local_training_data, local_test_data, local_sample_number, incremental_train_data,incremental_test_data,args, device,
                 model_trainer)
        
    def train(self, w_global):
        self.model_trainer.id = self.client_idx
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.set_model_global_params(w_global[0])

        if self.incremental_id == 0:
            self.model_trainer.train_first(self.local_training_data, self.incremental_id+1, self.device, self.args)
        else:
            self.model_trainer.train(self.local_training_data, self.incremental_id+1, self.device, self.args)
        weights = self.model_trainer.get_model_params()
        return weights