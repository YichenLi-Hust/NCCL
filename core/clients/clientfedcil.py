import logging
import random
import math
import copy
import torch
import numpy as np
import torch.nn.functional as F
import time
from utils.utils import transform_list_to_tensor
from core.clients.clienttarget_base import Client

class Clientfedcil(Client):
    
    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, incremental_train_data,incremental_test_data,args, device,
                 model_trainer):
        super().__init__(self, client_idx, local_training_data, local_test_data, local_sample_number, incremental_train_data,incremental_test_data,args, device,
                 model_trainer)
        self.init_local_training_data = local_training_data
        self.init_incremental_train_data = incremental_train_data
        self.init_incremental_test_data = incremental_test_data
        self.get_lable_data()
    
    def get_lable_data(self):
        data,unlable_num = self.get_data_base_ratio(self.init_local_training_data)
        self.local_training_data = data
        self.local_sample_number = self.local_sample_number - unlable_num

        self.incremental_train_data = []
        for i in range(len(self.init_incremental_train_data)):
            data,_ = self.get_data_base_ratio(self.init_incremental_train_data[i])
            self.incremental_train_data.append(data)
    
    def get_data_base_ratio(self,init_data):
        R=self.args.lable_ratio
        num= self.args.batch_size*(len(init_data)-1) + len(init_data[-1][0])
        unlable_num = (1-R)*num
        unlable_num = math.floor(unlable_num)
        np.random.seed(123)  
        un_id =  np.random.choice( num, unlable_num,False)
        # self.local_sample_number = self.local_sample_number - unlable_num

        new_train = []
        # new_train.append([])
        current_batch_id = -1
        current_batch_data_num = self.args.batch_size +5 #超出阈值，使得在初始时调用else分支

        for i in range(len(init_data)):#batch  然后data[0][0]是x，data[0][1]是y
            for j in range(len(init_data[i][1])):#batch中数据个数
                data_id = i*self.args.batch_size+j
                if data_id not in un_id:
                    temp_x=init_data[i][0][j]
                    temp_y=init_data[i][1][j]
                    temp_x= temp_x.unsqueeze(0)
                    temp_y= temp_y.unsqueeze(0)
                    if current_batch_data_num < self.args.batch_size:    
                        new_train[current_batch_id][0]=torch.cat((new_train[current_batch_id][0],temp_x),dim=0)
                        new_train[current_batch_id][1]=torch.cat((new_train[current_batch_id][1],temp_y),dim=0)
                        current_batch_data_num+=1
                    else:
                        new_train.append([])
                        current_batch_data_num = 1
                        current_batch_id += 1
                        new_train[current_batch_id]=[temp_x,temp_y]

        return new_train,unlable_num
    
    def update_incremental(self, w_global):

        if self.incremental_id < len(self.incremental_train_data):

            num_il = self.args.batch_size*(len(self.incremental_train_data[self.incremental_id])-1) + len(self.incremental_train_data[self.incremental_id][-1][0])
            #最后一个批次可能不满，num_il代表第incremental_id份增连数据的样本量
            print("Client: " + str(self.client_idx) + " will increase " + str(num_il) + " data samples")
            self.model_trainer.set_model_params(w_global)

            self.local_sample_number = num_il 
            self.local_training_data = self.incremental_train_data[self.incremental_id]
            self.local_test_data =  self.local_test_data + self.incremental_test_data[self.incremental_id]

            self.model_trainer.set_model_g_previous_params(w_global[0])
            #两个模型，g+d
            self.incremental_id += 1

        else:
            print("Client: " + str(self.client_idx) + " has no more incremental dataset")

    def train(self, w_global,ta_id,used_B):
        self.model_trainer.id = self.client_idx
        self.model_trainer.set_model_params(w_global)

        new_B = used_B
        num = (self.args.B / self.args.batch_size)* self.args.epochs * self.args.incremental_round
        num = math.floor(num)  
        if used_B < num:
            new_B=self.model_trainer.train(self.local_training_data, self.device, self.args,ta_id+1,used_B)
            # self.model_trainer.train(self.local_training_data, self.incremental_id+1, self.device, self.args)
        else:
            print('-----Client '+str(self.client_idx)+' no train')
        weights = self.model_trainer.get_model_params()

        return weights,new_B
    
    def test_update_incremental(self,num):
        self.local_test_data = self.init_test_data
        for i in range(num):
            self.local_test_data =  self.local_test_data+self.incremental_test_data[i]