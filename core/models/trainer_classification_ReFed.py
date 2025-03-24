import logging
import traceback
import copy
import torch
import math
from torch import nn
import torch.nn.functional as F
import numpy as np
# import wandb

from model_trainer import ModelTrainer
from core.models.trainer_classification_target_base import MyModelTrainer 

class MyModelTrainer_ReFed(MyModelTrainer):
    def __init__(self, model,args=None):
        self.model = model
        self.model_p = copy.deepcopy(model)
        self.args = args

    def get_model_params_p(self):
        return self.model_p.cpu().state_dict()

    def set_model_params_p(self,model_parameters):
        self.model_p.load_state_dict(model_parameters)

    def train_prox(self, train_data, device, args):
        logging.debug("-------model actually train------")
        try:
            model = self.model
            mu = 0.5

            model.to(device)
            model.train()
            global_model = copy.deepcopy(model)

            # train and update
            criterion = nn.CrossEntropyLoss().to(device)
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

                    x, labels = x.to(device), labels.to(device)
                    model.zero_grad()
                    log_probs = model(x)

                    proximal_term = 0.0
                    for w, w_t in zip(model.parameters(), global_model.parameters()):
                        proximal_term += (w - w_t).norm(2)

                    loss = criterion(log_probs, labels) + (mu / 2) * proximal_term

                    loss.backward()
                    batch_loss.append(loss.item())
                    optimizer.step()                 
                    
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                scheduler.step(sum(epoch_loss) / len(epoch_loss))
            logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
                    self.id, epoch, sum(epoch_loss) / len(epoch_loss)))

        except Exception as e:
            logging.error(traceback.format_exc())
        
    def train(self, train_data, device, args):
        logging.debug("-------model actually train------")
        try:
            model = self.model

            model.to(device)
            model.train()

            # train and update
            criterion = nn.CrossEntropyLoss().to(device)
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

                    x, labels = x.to(device), labels.to(device)
                    model.zero_grad()
                    log_probs = model(x)

                    loss = criterion(log_probs, labels)
                    loss.backward()
                    batch_loss.append(loss.item())
                    optimizer.step()                 
                    
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                scheduler.step(sum(epoch_loss) / len(epoch_loss))
            logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
                    self.id, epoch, sum(epoch_loss) / len(epoch_loss)))

        except Exception as e:
            logging.error(traceback.format_exc())

    def train_personal(self, train_data, device, args):
        logging.debug("-------gan model actually train------")
        try:
            model_p = self.model_p
            model_p.to(device)
            model_p.train()

            model = self.model
            model.to(device)
            
            importance_score = {}

            mu = arg.lambda_p

            # optimizer
            criterion = nn.CrossEntropyLoss().to(device)

            if args.client_optimizer == "sgd":
                d_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model_p.parameters()), lr=args.lr)
            else:
                d_optimizer = torch.optim.Adam(self.model_p.parameters(), lr=args.lr_p)
                                        
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(d_optimizer,
                                                           mode="min",
                                                           factor=0.5,
                                                           patience=2)

            # train and update
            img_num_list = []
            epoch_loss = []

            for epoch in range(args.epochs_personal):

                score_lst = []

                for(img, labels) in train_data:
                    batch_loss = []
                    num_img = img.size(0)
                    if epoch == 0:
                        img_num_list.append(num_img)
                    
                    # 训练分类
                    img, labels = img.to(device), labels.to(device)
                    model_p.zero_grad()
                    log_probs = model_p(img)

                    proximal_term = 0.0
                    for w, w_t in zip(model_p.parameters(), model.parameters()):
                        proximal_term += (w - w_t).norm(2)

                    loss = criterion(log_probs, labels) + (1-mu) / (2*mu) * proximal_term
                    batch_loss.append(loss.item())

                    loss.backward(retain_graph=True)
                    d_optimizer.step()

                    batch_score = 0

                    for i in range(num_img):
                        model_p.zero_grad()
                        sample_loss = criterion(log_probs[i], labels[i])
                        sample_loss.backward(retain_graph=True)

                        total_norm = 0
                        for p in model_p.parameters():
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** (1. / 2)

                        batch_score += total_norm

                    score_lst.append(batch_score)

                for i in range(len(score_lst)):
                    importance_score[i] = importance_score.get(i,0) + (((epoch+1)*(math.log(args.epochs_personal)+0.5772)))*score_lst[i]

                    # importance_score[i] = importance_score.get(i,0) + (1/((epoch+1)*(math.log(args.epochs_personal)+0.5772)))*score_lst[i]

                     
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                scheduler.step(sum(epoch_loss) / len(epoch_loss))

                if epoch == args.epochs_personal-1:
                    logging.info(
                    'Personal Model\tEpoch: {}\timg_nums: {}\tEpoch_Loss: {:.6f}'.format(
                      epoch, sum(img_num_list),sum(batch_loss) / len(batch_loss)))


        except Exception as e:
            logging.error(traceback.format_exc())

        return importance_score
    
    