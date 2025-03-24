import logging
import random
import math
import copy
import torch
import numpy as np
import torch.nn.functional as F

class Client(object):

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, incremental_train_data,incremental_test_data,args, device,
                 model_trainer):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.incremental_train_data = incremental_train_data
        self.incremental_test_data = incremental_test_data
        logging.info("self.local_sample_number = " + str(self.local_sample_number))
        self.args = args
        self.device = device
        self.model_trainer = model_trainer
        self.incremental_id = 0
        self.personal_model = self.model_trainer.get_model_params()      

    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def get_sample_number(self):
        return self.local_sample_number
        
    def update_incremental(self, w_global):

        if self.incremental_id < len(self.incremental_train_data):

            num_il = self.args.batch_size*(len(self.incremental_train_data[self.incremental_id])-1) + len(self.incremental_train_data[self.incremental_id][-1][0])
            logging.info("Client: " + str(self.client_idx) + " will increase " + str(num_il) + " data samples")
            self.model_trainer.set_model_params(w_global)

            self.local_sample_number = num_il

            self.local_training_data = self.incremental_train_data[self.incremental_id]
            self.local_test_data =  self.local_test_data + self.incremental_test_data[self.incremental_id]

            self.incremental_id += 1

        else:
            logging.info("Client: " + str(self.client_idx) + " has no more incremental dataset")
    
    def train(self, w_global,synthetic_data):
        self.model_trainer.id = self.client_idx
        self.model_trainer.set_model_params(w_global)

        self.model_trainer.train(self.local_training_data, synthetic_data, self.device, self.args)
        weights = self.model_trainer.get_model_params()


        return weights

    def local_test(self, b_use_test_dataset):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics