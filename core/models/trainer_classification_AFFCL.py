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

class MyModelTrainer_AFFCL(MyModelTrainer):
    def __init__(self, model,args=None):
        self.model = model[0]
        self.model_previous = copy.deepcopy(model[0])
        self.model_nf = model[1]
        self.model_nf_previous = copy.deepcopy(model[1])
        self.model_global = copy.deepcopy(model[0])
        self.args = args
        self.num_classes = 10
        self.eps = 1e-30
    
    def get_model_params(self):
        return (self.model.cpu().state_dict(),self.model_nf.cpu().state_dict())

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters[0])
        self.model_nf.load_state_dict(model_parameters[1])

    def set_model_previous_params(self, model_parameters):
        self.model_previous.load_state_dict(model_parameters[0])
        self.model_nf_previous.load_state_dict(model_parameters[1])

    def set_model_global_params(self, model_parameters):
        self.model_global.load_state_dict(model_parameters)
        

    def sample_from_flow(self, flow, labels, batch_size):
        label = np.random.choice(labels, batch_size)
        class_onehot = np.zeros((batch_size, self.num_classes))
        class_onehot[np.arange(batch_size), label] = 1
        class_onehot = torch.Tensor(class_onehot).cuda()
        flow_xa = flow.sample(num_samples=1, context=class_onehot).squeeze(1)
        flow_xa = flow_xa.detach()
        return flow_xa, label, class_onehot
    
    def MultiClassCrossEntropy(self, logits, labels, T):
        logits = torch.pow(logits+self.eps, 1/T)
        logits = logits/(torch.sum(logits, dim=1, keepdim=True)+self.eps)
        labels = torch.pow(labels+self.eps, 1/T)
        labels = labels/(torch.sum(labels, dim=1, keepdim=True)+self.eps)

        outputs = torch.log(logits+self.eps)
        outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
        outputs = -torch.mean(outputs, dim=0, keepdim=False)
        return outputs

    def probability_in_localdata(self, xa_u, y, prob_mean, flow_xa, flow_label):
        flow_xa_label_set = set(flow_label)
        flow_xa_prob = torch.zeros([flow_xa.shape[0]], device=flow_xa.device)
        for flow_yi in flow_xa_label_set:
            if (y==flow_yi).sum()>0:
                xa_u_yi = xa_u[y==flow_yi]
                xa_u_yi_mean = torch.mean(xa_u_yi, dim=0, keepdim=True)
                xa_u_yi_var = torch.mean((xa_u_yi-xa_u_yi_mean)*(xa_u_yi-xa_u_yi_mean), dim=0, keepdim=True)

                flow_xa_yi = flow_xa[flow_label==flow_yi]
                prob_xa_yi_ = 1/np.sqrt(2*np.pi)*torch.pow(xa_u_yi_var+self.eps, -0.5)*torch.exp(-torch.pow(flow_xa_yi-xa_u_yi_mean, 2)*torch.pow(xa_u_yi_var+self.eps, -1)*0.5)
                prob_xa_yi = torch.mean(prob_xa_yi_, dim=1)
                flow_xa_prob[flow_label==flow_yi] = prob_xa_yi
            else:
                flow_xa_prob[flow_label==flow_yi] = prob_mean
        return flow_xa_prob

    def knowledge_distillation_on_output(self, xa, softmax_output, last_classifier, global_classifier):
        if type(last_classifier)!=type(None):
            softmax_output_last, _ = last_classifier.forward_from_xa(xa)
            softmax_output_last = softmax_output_last.detach()
            kd_loss_output_last = 0.2 * self.MultiClassCrossEntropy(softmax_output, softmax_output_last, T=2)
        else:
            kd_loss_output_last = 0

        softmax_output_global, _ = global_classifier.forward_from_xa(xa)
        softmax_output_global = softmax_output_global.detach()
        kd_loss_output_global = 0.2 * self.MultiClassCrossEntropy(softmax_output, softmax_output_global, T=2)


        return kd_loss_output_last, kd_loss_output_global
    
    def knowledge_distillation_on_xa_output(self, x, xa, softmax_output, last_classifier, global_classifier):
        if type(last_classifier)!=type(None):
            softmax_output_last, xa_last, _ = last_classifier(x)
            xa_last = xa_last.detach()
            softmax_output_last = softmax_output_last.detach()
            kd_loss_feature_last = 0.2 * torch.pow(xa_last-xa, 2).mean()
            kd_loss_output_last = 0.2 * self.MultiClassCrossEntropy(softmax_output, softmax_output_last, T=2)
        else: 
            kd_loss_feature_last = 0
            kd_loss_output_last = 0

        softmax_output_global, xa_global, _ = global_classifier(x)
        xa_global = xa_global.detach()
        softmax_output_global = softmax_output_global.detach()
        kd_loss_feature_global = 0.2 * torch.pow(xa_global-xa, 2).mean()
        kd_loss_output_global = 0.2 * self.MultiClassCrossEntropy(softmax_output, softmax_output_global, T=2)

        return kd_loss_feature_last, kd_loss_output_last, kd_loss_feature_global, kd_loss_output_global

    def train_first(self, train_data, id, device, args):
        logging.debug("-------model actually train------")
        try:
            model = self.model
            model_nf = self.model_nf

            model.to(device)
            model.train()
            model_nf.to(device)
            model_nf.train()

            # train and update
            criterion = nn.NLLLoss().to(device)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=5e-4)
            optimizer_nf = torch.optim.Adam(self.model_nf.parameters(), lr=args.lr, weight_decay=5e-4)

            logging.info("Train the NF model")
            for epoch in range(args.epochs):
                for batch_idx, (x, labels) in enumerate(train_data):
                    x, labels = x.to(device), labels.to(device)
                    xa = model.forward_to_xa(x)
                    xa = xa.reshape(xa.shape[0], -1)
                    y_one_hot = F.one_hot(labels, num_classes=10).float()
                    loss_data = -model_nf.log_prob(inputs=xa, context=y_one_hot).mean()
                    optimizer_nf.zero_grad()
                    loss_data.backward()
                    optimizer_nf.step()
            
            logging.info("Train the local model")
            for epoch in range(args.epochs):
                for batch_idx, (x, labels) in enumerate(train_data):
                    x, labels = x.to(device), labels.to(device)
                    softmax_output, xa, logits = model(x)
                    c_loss_cls = criterion(torch.log(softmax_output + self.eps), labels)
                    optimizer.zero_grad()
                    c_loss_cls.backward()
                    optimizer.step()

            logging.info('Client Index = {}\tEpoch: {}'.format(
                    self.id, epoch,))
            
        except Exception as e:
            logging.error(traceback.format_exc())

    def train(self, train_data, id, device, args):
        logging.debug("-------model actually train------")
        try:
            model = self.model
            model_nf = self.model_nf
            model_nf_previous = self.model_nf_previous
            model_previous = self.model_previous
            model_global = self.model_global

            model.to(device)
            model.train()
            model_nf.to(device)
            model_nf.train()
            model_nf_previous.eval()
            model_nf_previous.to(device)
            model_previous.eval()
            model_previous.to(device)
            model_global.eval()
            model_global.to(device)

            # train and update
            criterion = nn.NLLLoss().to(device)

            optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=5e-4)
            optimizer_nf = torch.optim.Adam(self.model_nf.parameters(), lr=args.lr, weight_decay=5e-4)

            parameters_fb = [a[1] for a in filter(lambda x: 'fc2' in x[0], self.model.named_parameters())]
            optimizer_fb = torch.optim.Adam(parameters_fb, lr=args.lr, weight_decay=5e-4)

            epoch_loss = []

            logging.info("Train the NF model")
            for epoch in range(args.epochs//2):
                for batch_idx, (x, labels) in enumerate(train_data):
                    x, labels = x.to(device), labels.to(device)
                    xa = model.forward_to_xa(x)
                    xa = xa.reshape(xa.shape[0], -1)
                    y_one_hot = F.one_hot(labels, num_classes=10).float()
                    loss_data = -model_nf.log_prob(inputs=xa, context=y_one_hot).mean()

                    batch_size = x.shape[0]
                    with torch.no_grad():
                        flow_xa, label, label_one_hot = self.sample_from_flow(model_nf_previous, 2*(id-1), batch_size)

                    loss_last_flow = -model_nf.log_prob(inputs=flow_xa, context=label_one_hot).mean()

                    loss_last_flow = 0.4 * loss_last_flow

                    loss = loss_data + loss_last_flow

                    optimizer_nf.zero_grad()
                    loss.backward()
                    optimizer_nf.step()
            
            logging.info("Train the local model")
            for epoch in range(args.epochs):
                for batch_idx, (x, labels) in enumerate(train_data):
                    x, labels = x.to(device), labels.to(device)
                    with torch.no_grad():
                        _, xa, _ = model(x)
                        xa = xa.reshape(xa.shape[0], -1)

                        y_one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
                        log_prob, xa_u = model_nf.log_prob_and_noise(xa, y_one_hot)
                        log_prob = log_prob.detach()
                        xa_u = xa_u.detach()
                        prob_mean = torch.exp(log_prob/xa.shape[1]).mean() + self.eps

                        flow_xa, label, _ = self.sample_from_flow(model_nf, 2*id, batch_size)
                        flow_xa_prob = self.probability_in_localdata(xa_u, labels, prob_mean, flow_xa, label)
                        flow_xa_prob = flow_xa_prob.detach()
                        flow_xa_prob_mean = flow_xa_prob.mean()

                    flow_xa = flow_xa.reshape(flow_xa.shape[0], *[512])
                    softmax_output_flow, _ = model.forward_from_xa(flow_xa)
                    c_loss_flow_generate = (criterion(torch.log(softmax_output_flow+self.eps), torch.Tensor(label).long().cuda())*flow_xa_prob).mean()
                    k_loss_flow_explore_forget = (1-0.2) * prob_mean + 0.2

                    kd_loss_output_last_flow, kd_loss_output_global_flow = self.knowledge_distillation_on_output(flow_xa, softmax_output_flow, model_previous, model_global)
                    kd_loss_flow = (kd_loss_output_last_flow + kd_loss_output_global_flow) * 0.1

                    c_loss_flow = (c_loss_flow_generate*k_loss_flow_explore_forget + kd_loss_flow) * 0.1
                    
                    optimizer_fb.zero_grad()
                    c_loss_flow.backward()
                    optimizer_fb.step()

                    softmax_output, xa, logits = model(x)
                    
                    c_loss_cls = criterion(torch.log(softmax_output + self.eps), labels)


                    kd_loss_feature_last, kd_loss_output_last, kd_loss_feature_global, kd_loss_output_global = \
                                                self.knowledge_distillation_on_xa_output(x, xa, softmax_output, model_previous, model_global)
                    
                    kd_loss_feature = (kd_loss_feature_last + kd_loss_feature_global) * 0.5

                    kd_loss_output = (kd_loss_output_last + kd_loss_output_global) * 0.1

                    kd_loss = kd_loss_feature + kd_loss_output


                    c_loss = c_loss_cls + kd_loss

                    optimizer.zero_grad()
                    c_loss.backward()
                    optimizer.step()            
                    

            logging.info('Client Index = {}\tEpoch: {}'.format(
                    self.id, epoch))

        except Exception as e:
            logging.error(traceback.format_exc())

    