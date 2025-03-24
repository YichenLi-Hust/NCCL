import logging
import traceback
import copy
import torch
import math
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
# import wandb

from model_trainer import ModelTrainer


class MyModelTrainer(ModelTrainer):
    def __init__(self, model,args=None):
        self.model = model[0]
        self.model_g = model[1]
        self.model_previous = copy.deepcopy(model[0])
        self.model_s = copy.deepcopy(model[0])
        self.args = args

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def set_model_previous_params(self, model_parameters):
        self.model_previous.load_state_dict(model_parameters)

    def get_synthetic_data(self,id,device):
        model_g = self.model_g
        model_g.to(device)
        model_previous = self.model_previous
        model_previous.to(device)
        distillation_share_data = []
    
        with torch.no_grad():
            for i in range(self.args.batch_size, self.args.M, self.args.batch_size):
                input_label = Variable(torch.randint(0, id*2, (self.args.batch_size,))).to(device)
                noise = Variable(torch.randn(self.args.batch_size, self.args.noise_dimension)).to(device)
                batch_share_data = model_g(noise)
                label = model_previous(batch_share_data)
                distillation_share_data.append((batch_share_data,label))
        return distillation_share_data
    
    def global_consilidation(self,id,device):
        if id > 1:
            logging.info('Start the Global Training!')
            model_g = self.model_g
            model_previous = self.model_previous
            model_s = self.model_s
            model_g.to(device)
            model_previous.to(device)
            model_s.to(device)

            criterion = nn.CrossEntropyLoss().to(device)
            criterion2 = nn.KLDivLoss()
            T = 20

            betas = (0.5, 0.99)

            g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model_g.parameters()), lr=self.args.lr,
                                               betas=betas, amsgrad=True)
            s_optimizer = torch.optim.Adam(self.model_s.parameters(), lr=self.args.lr)
                                               

            for epoch in range(20*self.args.epochs):

                input_label = Variable(torch.randint(0, id*2, (self.args.batch_size,))).to(device)
                noise = Variable(torch.randn(self.args.batch_size, self.args.noise_dimension)).to(device)
                batch_share_data = model_g(noise)

                lob_previous = model_previous(batch_share_data)

                loss_ce = criterion(lob_previous,input_label)

                lob_s = model_s(batch_share_data)

                loss_kl = (T ** 2) * criterion2(
                                torch.nn.functional.log_softmax(lob_s / T, dim=1),
                                torch.nn.functional.softmax(lob_previous / T, dim=1)
                )

                loss_all = loss_ce + loss_kl

                loss_all.backward()
                g_optimizer.step()

                if epoch > self.args.epochs//3:
                    noise = Variable(torch.randn(self.args.batch_size, self.args.noise_dimension)).to(device)
                    batch_share_data = model_g(noise)

                    lob_previous = model_previous(batch_share_data)
                    lob_s = model_s(batch_share_data)

                    loss_kl = (T ** 2) * criterion2(
                                torch.nn.functional.log_softmax(lob_s / T, dim=1),
                                torch.nn.functional.softmax(lob_previous / T, dim=1)
                    )

                    loss_kl.backward()
                    s_optimizer.step()
            logging.info('Finish the Global Training!')


            return True

        else:
            return False

    def train(self, train_data, synthetic_data, device, args):
        logging.debug("-------model actually train------")
        try:
            model = self.model

            model.to(device)
            model.train()

            # train and update
            criterion = nn.CrossEntropyLoss().to(device)
            criterion2 = nn.KLDivLoss()
            T = 20

            if args.client_optimizer == "sgd":
                optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
            else:
                optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=5e-4)

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode="min",
                                                           factor=0.2,
                                                           patience=2)
            epoch_loss = []
            for epoch in range(args.epochs):
                batch_loss = []
                for batch_idx, (x, labels) in enumerate(train_data):
                    num_img = len(labels)

                    x, labels = x.to(device), labels.to(device)
                    model.zero_grad()
                    log_probs = model(x)

                    loss = criterion(log_probs, labels)

                    if synthetic_data != []:
                        random_data = random.sample(synthetic_data[:-1],1)[0]
                        epoch_data  = random_data[0][:2*num_img].to(device)
                        epoch_label = random_data[1][:2*num_img].to(device)
                        pre = model(epoch_data)
                        loss_kl = (T ** 2) * criterion2(
                                torch.nn.functional.log_softmax(pre / T, dim=1),
                                torch.nn.functional.softmax(epoch_label / T, dim=1)
                        )
                        # print(loss.item(),loss_kl.item())
                        loss += 20*loss_kl

                    loss.backward()
                    batch_loss.append(loss.item())
                    optimizer.step()                 
                    
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                scheduler.step(sum(epoch_loss) / len(epoch_loss))
            logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
                    self.id, epoch, sum(epoch_loss) / len(epoch_loss)))

        except Exception as e:
            logging.error(traceback.format_exc())

    def test(self, test_data, device, args):

        def calculate_top_k_accuracy(logits, targets, k=5):
            correct = 0
            values, indices = torch.topk(logits, k=k, sorted=True)
            for i in range(len(targets)):
                if targets[i] in indices[i]:
                    correct += 1
            return correct
        
        model = self.model
        model.to(device)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0
        }

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()
                metrics['test_correct'] += correct.item()

                # metrics['test_correct'] += calculate_top_k_accuracy(pred,target,10)
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)

        return metrics
    

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False