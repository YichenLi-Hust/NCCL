import copy
import logging
import random

import numpy as np
import torch
from utils.utils import transform_list_to_tensor
from core.clients.clienttarget_base import Client


class Server(object):
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
        self.task_id = 1
        self._setup_clients(local_num_dict, train_data_local_dict, test_data_local_dict, incremental_train_data,incremental_test_data,model_trainer)

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict,incremental_train_data,incremental_test_data,model_trainer):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_in_total):
            # Test with all samples
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict, train_data_local_num_dict[client_idx], 
                                        incremental_train_data[client_idx], incremental_test_data[client_idx], self.args, self.device, model_trainer)
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def train(self):

        flag = False

        for round_idx in range(self.args.comm_round):

            if flag:
                synthetic_data = self.model_trainer.get_synthetic_data(self.task_id,self.device)
            else:
                synthetic_data = []

            w_global = self.model_trainer.get_model_params()

            logging.info("################Communication round : {}".format(round_idx))
            w_locals = []

            # incremental learning (update)
            if round_idx % self.args.incremental_round == 0 and round_idx != 0:
                logging.info("Start updating each client dataset by incremental learning! " )
                for client in self.client_list:
                    client.update_incremental(w_global)
                self.task_id += 1
                self.model_trainer.set_model_previous_params(w_global)
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
                weight = client.train(copy.deepcopy(w_global), synthetic_data)
                w_locals.append((client.get_sample_number(), copy.deepcopy(weight)))
               
            w_global = self._aggregate(w_locals)

            flag = self.model_trainer.global_consilidation(self.task_id,self.device)



            if round_idx % 1 == 0:
                self._local_test_on_all_clients(round_idx)



    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round,client_list):
        if client_num_in_total == client_num_per_round:
            self.client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  
            self.client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)

    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w

        global_model_params = averaged_params

        self.model_trainer.set_model_params(global_model_params)

        return global_model_params

    def _local_test_on_all_clients(self, round_idx):

        logging.info("################local_test_on_all_clients : {}".format(round_idx))

        train_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        # client = self.client_list[0]

        for client in self.client_list:
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """

            train_local_metrics = client.local_test(False)
            train_metrics['num_samples'].append(copy.deepcopy(train_local_metrics['test_total']))
            train_metrics['num_correct'].append(copy.deepcopy(train_local_metrics['test_correct']))
            train_metrics['losses'].append(copy.deepcopy(train_local_metrics['test_loss']))

            # test data
            test_local_metrics = client.local_test(True)
            test_metrics['num_samples'].append(copy.deepcopy(test_local_metrics['test_total']))
            test_metrics['num_correct'].append(copy.deepcopy(test_local_metrics['test_correct']))
            test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))


        # test on training dataset
        train_acc = sum(train_metrics['num_correct']) / sum(train_metrics['num_samples'])
        train_loss = sum(train_metrics['losses']) / sum(train_metrics['num_samples'])

        test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])

        stats = {'training_acc': train_acc, 'training_loss': train_loss}
        # wandb.log({"Train/Acc": train_acc, "round": round_idx})
        # wandb.log({"Train/Loss": train_loss, "round": round_idx})
        logging.info(stats)

        stats = {'test_acc': test_acc, 'test_loss': test_loss}
        # wandb.log({"Test/Acc": test_acc, "round": round_idx})
        # wandb.log({"Test/Loss": test_loss, "round": round_idx})
        logging.info(stats)
        self.train_acc.append(train_acc)
        self.test_acc.append(test_acc)


