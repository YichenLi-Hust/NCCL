import logging
import random
import math
import copy
import torch
import numpy as np
import torch.nn.functional as F
from core.clients.clienttarget_base import Client

class Clientrefed(Client):

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, incremental_train_data,incremental_test_data,args, device,
                 model_trainer):
        super().__init__(self, client_idx, local_training_data, local_test_data, local_sample_number, incremental_train_data,incremental_test_data,args, device,
                 model_trainer)
        
    def update_incremental(self, w_global):

        if self.incremental_id < len(self.incremental_train_data):

            num_il = self.args.batch_size*(len(self.incremental_train_data[self.incremental_id])-1) + len(self.incremental_train_data[self.incremental_id][-1][0])
            logging.info("Client: " + str(self.client_idx) + " will increase " + str(num_il) + " data samples")
            delete_num = max(num_il + self.local_sample_number - self.args.memory_size, 0)

            self.model_trainer.set_model_params_p(copy.deepcopy(self.personal_model))


            self.model_trainer.set_model_params(w_global)

            # if delete_num < 0 : # wrong here
            if delete_num > 0 : 

                self.local_sample_number = self.args.memory_size

                importance_score = self.model_trainer.train_personal(self.local_training_data, self.device, self.args)
                sorted_items = sorted(importance_score.items(), key=lambda item: item[1], reverse=False)
                delete_samples = [item[0] for item in sorted_items[:delete_num]]

                # if delete_num < self.args.batch_size:
                #     delete_samples = [0]
                # else:
                #     delete_samples = [i for i in range(math.ceil(delete_num//self.args.batch_size))]

                temp = []
                for i in range(len(self.local_training_data)):
                    if i not in delete_samples:
                        temp.append(self.local_training_data[i])
                
                self.local_training_data = temp + self.incremental_train_data[self.incremental_id]
                self.local_test_data += self.incremental_test_data[self.incremental_id]


            else:
                self.local_sample_number = num_il + self.local_sample_number

                # forgetting
                # self.local_training_data = self.incremental_train_data[self.incremental_id]


                self.local_training_data += self.incremental_train_data[self.incremental_id]
                self.local_test_data =  self.local_test_data + self.incremental_test_data[self.incremental_id]

            self.incremental_id += 1

            # if self.client_idx == 0:
            #     print(self.local_test_data)

        else:
            logging.info("Client: " + str(self.client_idx) + " has no more incremental dataset")
    
    def train(self, w_global):
        self.model_trainer.id = self.client_idx
        self.model_trainer.set_model_params(w_global)

        # self.model_trainer.train_prox(self.local_training_data, self.device, self.args)


        self.model_trainer.train(self.local_training_data, self.device, self.args)
        weights = self.model_trainer.get_model_params()

        # print(len(self.local_test_data))

        return weights

