import copy
import logging
import random
import time
import numpy as np
import torch
from utils.utils import transform_list_to_tensor
from core.clients.clientfedcil import Clientfedcil

from core.servers.servertarget_base import Server

class FedCIL(Server):
    def __init__(self, dataset, device, args, model_trainer):
        super().__init__(self, dataset, device, args, model_trainer)       

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict,incremental_train_data,incremental_test_data,model_trainer):
        print("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_in_total):
            # Test with all samples
            c = Clientfedcil(client_idx, train_data_local_dict[client_idx], test_data_local_dict, train_data_local_num_dict[client_idx], 
                                        incremental_train_data[client_idx], incremental_test_data[client_idx], self.args, self.device, model_trainer)
            self.client_list.append(c)
        print("############setup_clients (END)#############")

    def train(self,savepath):

        start_time = time.time()
        task_used_B=[]
        for round_idx in range(self.args.comm_round):
            
            node1_time = time.time()
            print("到达round: "+str(round_idx)+" 耗时:"+str(node1_time - start_time)+"秒")

            w_global = self.model_trainer.get_model_params()

            print("################Communication round : {}".format(round_idx))
            w_locals = []

            # 计算当前任务中的当前轮数  
            current_round_in_task = round_idx % self.args.incremental_round 
            ta_id = round_idx // self.args.incremental_round 

            # incremental learning (update)
            if round_idx % self.args.incremental_round == 0 and round_idx != 0:
                print("Start updating each client dataset by incremental learning! " )
                for client in self.client_list:
                    client.update_incremental(w_global)
                self.task_id += 1
                print("Finishing updating! " )

            #选择客户端
            self._client_sampling(round_idx, self.args.client_num_in_total,
                                                   self.args.client_num_per_round,self.client_list)
                                                   
            print("client_indexes = " + str(self.client_indexes))

            # choose client 训练
            if current_round_in_task == 0:#新任务
                task_used_B= [0] * self.args.client_num_in_total

            for idx in self.client_indexes:#w每次用全局模型聚合后的还是聚合前每个客户端自己的
                client_idx = idx
                for i in self.client_list:
                    if i.client_idx == client_idx:
                        client = i

                weight,new_B = client.train(copy.deepcopy(w_global),ta_id,task_used_B[client_idx])
                task_used_B[client_idx]=new_B
                w_locals.append((client.get_sample_number(), copy.deepcopy(weight)))
            w_global = self._aggregate(w_locals)
            
            #add model_consolidation
            self.model_trainer.model_consolidation(self.task_id,self.device)

            if round_idx % 1 == 0:
                self._local_test_on_all_clients(round_idx)
                print(task_used_B)
            # if round_idx % 1 == 0:
            #     f = open(str(self.args.dataset)+"_re/fdil_lateEmphsis"+str(self.args.dataset)+"_a="+str(self.args.alpha)+"_lambda_"+str(self.args.lambda_p)+"_model_"+str(self.args.model)+"_lr="+str(self.args.lr)+"_size="+str(self.args.memory_size)+".txt",'w')
            #     for i in range(len(self.train_acc)):
            #         f.write("train acc:"+str(self.train_acc[i])+" "+"test acc:"+str(self.test_acc[i])+'\n')
            #     f.close()
            #     print( str(round_idx)+": train acc:"+str(self.train_acc[i])+" "+"test acc:"+str(self.test_acc[i])+'\n')


            if (self.args.incremental_round - 10) <= current_round_in_task < self.args.incremental_round:
                pa = self.model_trainer.get_model_params()
                torch.save(pa,savepath+'/model_%d_%d.pth'%(ta_id,current_round_in_task))
        end_time = time.time()
        print("end耗时:"+str(end_time - start_time)+"秒")

    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        # 更新初步集成模型
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
    
    def _local_test_on_all_clients(self, round_idx):

        print("################local_test_on_all_clients : {}".format(round_idx))

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
        print(stats)

        stats = {'test_acc': test_acc, 'test_loss': test_loss}
        # wandb.log({"Test/Acc": test_acc, "round": round_idx})
        # wandb.log({"Test/Loss": test_loss, "round": round_idx})
        print(stats)
        self.train_acc.append(train_acc)
        self.test_acc.append(test_acc)

    def test(self,num,savepath):

        model_pa = torch.load(savepath)
        self.model_trainer.set_model_params(model_pa)
        ### 选择参与训练的客户端（self.args.client_num_per_round个），索引保存在self.client_indexes
        # \(client_idx, local_training_data, local_test_data, local_sample_number, incremental_train_data,incremental_test_data,args, device, model_trainer):
        # client = Client(0, 'none','none',0,'none','none', self.args,self.device,self.model_trainer)
        # client= Client(0, train_data_local_dict[0], test_data_local_dict, train_data_local_num_dict[0], 
        #                                 incremental_train_data[0], incremental_test_data[0], self.args, self.device, model_trainer)
        client = self.client_list[0]
        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }
        result_acc=[]
        client.test_update_incremental(num)
        test_local_metrics = client.local_test(True)
        test_metrics['num_samples'].append(copy.deepcopy(test_local_metrics['test_total']))
        test_metrics['num_correct'].append(copy.deepcopy(test_local_metrics['test_correct']))
        test_acc = test_local_metrics['test_correct'] /test_local_metrics['test_total']
        # test_acc_total = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        stats = {'test_acc': test_acc}
        print(stats)
        result_acc.append(test_acc)
        # for i in range(num):
        #     client.test_update_incremental(i)
        #     test_local_metrics = client.local_test(True)
        #     test_metrics['num_samples'].append(copy.deepcopy(test_local_metrics['test_total']))
        #     test_metrics['num_correct'].append(copy.deepcopy(test_local_metrics['test_correct']))
        #     test_acc = test_local_metrics['test_correct'] /test_local_metrics['test_total']
        #     test_acc_total = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        #     stats = {'test_acc': test_acc, 'test_acc_total': test_acc_total}
        #     print(stats)
        #     result_acc.append(test_acc)
        print('-----------------------------------------')
        return result_acc

