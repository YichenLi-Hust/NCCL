import copy
import logging
import random

import numpy as np
import torch
from utils.utils import transform_list_to_tensor
from core.clients.clientaffcl import Clientaffcl

from core.servers.servertarget_base import Server

class AFFCL(Server):
    def __init__(self, dataset, device, args, model_trainer):
        self.device = device
        self.args = args
        [local_num_dict, train_data_local_dict, test_data_local_dict, \
        incremental_train_data, incremental_test_data, class_num] = dataset
        self.client_indexes = []
        self.client_list = []
        self.incremental_train_data = incremental_train_data
        self.incremental_test_data = incremental_test_data
        self.train_data_local_num_dict = local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.model_dict = dict()
        self.model_trainer = model_trainer
        self.train_acc = []
        self.test_acc = []
        self._setup_clients(local_num_dict, train_data_local_dict, test_data_local_dict, incremental_train_data,incremental_test_data,model_trainer)

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict,incremental_train_data,incremental_test_data,model_trainer):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_in_total):
            # Test with all samples
            c = Clientaffcl(client_idx, train_data_local_dict[client_idx], test_data_local_dict, train_data_local_num_dict[client_idx], 
                                        incremental_train_data[client_idx], incremental_test_data[client_idx], self.args, self.device, model_trainer)
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def train(self):

        for round_idx in range(self.args.comm_round):

            w_global = self.model_trainer.get_model_params()

            logging.info("################Communication round : {}".format(round_idx))
            w_locals = []

            # incremental learning (update)
            if round_idx % self.args.incremental_round == 0 and round_idx != 0:
                logging.info("Start updating each client dataset by incremental learning! " )
                for client in self.client_list:
                    client.update_incremental(w_global)

                logging.info("Finishing updating! " )

            self._client_sampling(round_idx, self.args.client_num_in_total,
                                                   self.args.client_num_per_round,self.client_list)
            logging.info("client_indexes = " + str(self.client_indexes))


            # choose client
            for idx in self.client_indexes:
                client_idx = idx
                for i in self.client_list:
                    if i.client_idx == client_idx:
                        client = i

                # train on new dataset
                weight = client.train(copy.deepcopy(w_global))
                w_locals.append((client.get_sample_number(), copy.deepcopy(weight)))
               
            w_global = self._aggregate(w_locals)

            if round_idx % 1 == 0:
                self._local_test_on_all_clients(round_idx)

            # if round_idx % 1 == 0:
            #     f = open(str(self.args.dataset)+"_re/fdil_lateEmphsis"+str(self.args.dataset)+"_a="+str(self.args.alpha)+"_lambda_"+str(self.args.lambda_p)+"_model_"+str(self.args.model)+"_lr="+str(self.args.lr)+"_size="+str(self.args.memory_size)+".txt",'w')
            #     for i in range(len(self.train_acc)):
            #         f.write("train acc:"+str(self.train_acc[i])+" "+"test acc:"+str(self.test_acc[i])+'\n')
            #     f.close()

    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params[0].keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[0][k] = local_model_params[0][k] * w
                else:
                    averaged_params[0][k] += local_model_params[0][k] * w

        for k in averaged_params[1].keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[1][k] = local_model_params[1][k] * w
                else:
                    averaged_params[1][k] += local_model_params[1][k] * w

        global_model_params = averaged_params

        self.model_trainer.set_model_params(global_model_params)

        return global_model_params
    
    