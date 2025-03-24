import logging
import traceback
import copy
import torch
import math
import random
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
# import wandb

from model_trainer import ModelTrainer
from core.models.trainer_classification_target_base import ModelTrainer

class MyModelTrainer_FedCIL(ModelTrainer):
    def __init__(self, model,args=None):
        self.model_g = model[0]
        self.model_d = model[1]
        self.model_g_previous = copy.deepcopy(model[0])
        self.args = args

    def get_model_params(self):
        return (self.model_g.cpu().state_dict(),self.model_d.cpu().state_dict())

    def set_model_params(self, model_parameters):
        self.model_g.load_state_dict(model_parameters[0])
        self.model_d.load_state_dict(model_parameters[1])

    def  set_model_g_previous_params(self,model_parameter):
        self.model_g_previous.load_state_dict(model_parameter)

    def get_synthetic_data(self,id,device,flag=True):#只生成旧任务
        if flag:
            model_g = self.model_g
        else:
            model_g = self.model_g_previous
        

        model_g.to(device)
        distillation_share_data = []
        task_class = self.args.task_class
        with torch.no_grad():
            for i in range(self.args.batch_size, self.args.M, self.args.batch_size):
                input_label = Variable(torch.randint(0, id*task_class, (self.args.batch_size,))).to(device)
                noise = Variable(torch.randn(self.args.batch_size, self.args.noise_dimension)).to(device)
                batch_share_data = model_g(noise,input_label)
                distillation_share_data.append((batch_share_data,input_label))
        return distillation_share_data
    
    def model_consolidation(self,id,device):
        #全局上的增强
        print("-------model_consolidation------")
        try:
            model_g = self.model_g
            model_d = self.model_d
            model_g.to(device)
            model_d.to(device)

            # ########B
            # current_B=used_B
            # num = self.args.B/self.args.batch_size
            # num = (self.args.B / self.args.batch_size)* self.args.epochs *self.args.incremental_round
            # num = math.floor(num)  
            
            model_g.train()
            model_d.train()
            train_data = self.get_synthetic_data(id,device)

            noise_dimension = self.args.noise_dimension

            criterion = nn.CrossEntropyLoss().to(device)

            betas = (0.5, 0.99)

            g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model_g.parameters()), lr=self.args.lr,
                                               betas=betas, amsgrad=True)
            d_optimizer = torch.optim.Adam(self.model_d.parameters(), lr=self.args.lr)
                                               
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(d_optimizer,
                                                           mode="min",
                                                           factor=0.2,
                                                           patience=2)

            # train and update
            g_epoch_loss = []
            d_epoch_loss = []
            img_num_list = []
            epoch_loss = []

            for epoch in range(self.args.epochs):
                for(img, labels) in train_data:
                    batch_loss = []
                    g_batch_loss = []
                    d_batch_loss = []
                    
                    num_img = img.size(0)
                    if epoch == 0:
                        img_num_list.append(num_img)
                    

                    # 训练分类
                    img, labels = img.to(device), labels.to(device)
                    model_d.zero_grad()

                    log_probs,real_out = model_d(img)
                    loss1 = criterion(log_probs, labels)
                    batch_loss.append(loss1.item())

                    # 训练判别
                    real_label = Variable(torch.ones(num_img)).to(device)
                    real_out = real_out.flatten()
                    d_loss_real = criterion(real_out, real_label)
                    noise = Variable(torch.randn(num_img, noise_dimension)).to(device)
                    label = Variable(torch.randint(0, self.args.total_class, (num_img,))).to(device)

                    fake_img = model_g(noise,label).detach()

                    fake_label = Variable(torch.zeros(num_img)).to(device)
                    fake_out = model_d(fake_img)[1]
                    fake_out = fake_out.flatten()

                    d_loss_fake = criterion(fake_out, fake_label)
                    d_loss = d_loss_real + d_loss_fake 
                    d_batch_loss.append(d_loss.item())
                    loss_all = loss1+d_loss
                    loss_all.backward()
                    d_optimizer.step()


                    # 训练生成
                    noise = Variable(torch.randn(num_img, noise_dimension)).to(device)
                    fake_img = model_g(noise,label)
                    output = model_d(fake_img)[1]
                    output = output.flatten()
                    g_loss = criterion(output, real_label)
                    g_optimizer.zero_grad()
                    g_loss.backward()
                    g_optimizer.step()
                    g_batch_loss.append(g_loss.item())

                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                g_epoch_loss.append(sum(g_batch_loss) / len(g_batch_loss))
                d_epoch_loss.append(sum(d_batch_loss) / len(d_batch_loss))
                scheduler.step(sum(epoch_loss) / len(epoch_loss))
                    
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                scheduler.step(sum(epoch_loss) / len(epoch_loss))

            print('Finish Global Consolidation')
        

        except Exception as e:
            logging.error(traceback.format_exc())

    def train(self, train_data, device, args ,id,used_B):
        logging.debug("-------model actually train------")
        try:
            model_g = self.model_g
            model_d = self.model_d

            model_g.to(device)
            model_d.to(device)

            ########B
            current_B=used_B
            num = args.B/args.batch_size
            num = (args.B / args.batch_size)* args.epochs * args.incremental_round
            num = math.floor(num)  

            model_g.train()
            model_d.train()

            noise_dimension = args.noise_dimension

            criterion = nn.CrossEntropyLoss().to(device)

            criterion2 = nn.KLDivLoss()

            betas = (0.5, 0.99)
            T = 20

            g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model_g.parameters()), lr=args.lr,
                                               betas=betas, amsgrad=True)
            d_optimizer = torch.optim.Adam(self.model_d.parameters(), lr=args.lr)
                                               
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(d_optimizer,
                                                           mode="min",
                                                           factor=0.2,
                                                           patience=2)

            g_epoch_loss = []
            d_epoch_loss = []
            img_num_list = []
            epoch_loss = []
            if id > 1:
                previou_task_data = self.get_synthetic_data(id-1,device,flag=False)
                new_train_data = previou_task_data + train_data
            else:
                new_train_data = train_data
            
            if len(train_data[-1][0]) == 1:
                tt =  train_data[:-1]
                train_data = tt 
                
            if len(new_train_data[-1][0]) == 1:
                tt =  new_train_data[:-1]
                new_train_data = tt 


            for epoch in range(args.epochs):
                for(img, labels) in new_train_data:
                    batch_loss = []
                    g_batch_loss = []
                    d_batch_loss = []
                    
                    num_img = img.size(0)
                    if epoch == 0:
                        img_num_list.append(num_img)
                    

                    # 训练分类
                    img, labels = img.to(device), labels.to(device)
                    model_d.zero_grad()
    

                    log_probs,real_out = model_d(img)#torch.Size([64, 3, 32, 32])
                    real_out = real_out.flatten()
                    
                    loss1 = criterion(log_probs, labels)
                    batch_loss.append(loss1.item())

                    input_label = Variable(torch.randint(0, id*args.task_class, (num_img,))).to(device)
                    noise = Variable(torch.randn(num_img, noise_dimension)).to(device)

                    fake_img = model_g(noise,input_label).detach()

                    # 训练判别
                    real_label = Variable(torch.ones(num_img)).to(device)
                    
                    d_loss_real = criterion(real_out, real_label)

                    fake_label = Variable(torch.zeros(num_img)).to(device)
                    fake_out = model_d(fake_img)[1]
                    fake_out = fake_out.flatten()

                    d_loss_fake = criterion(fake_out, fake_label)
                    d_loss = d_loss_real + d_loss_fake 
                    d_batch_loss.append(d_loss.item())

                    def hard_2_logit(lst,num_classes=10):
                        logit = []
                        for i in range(len(lst)):
                            temp = [0]*num_classes
                            temp[lst[i]] = 1000
                            logit.append(temp)
                        return Variable(torch.tensor(logit))

                    if id > 1:

                        # c1
                        input_label = Variable(torch.randint(0, id*args.task_class, (num_img,))).to(device)
                        noise = Variable(torch.randn(num_img, noise_dimension)).to(device)
                        
                        c_g = model_g(noise,input_label).to(device)
                        c_g_previous = random.sample(previou_task_data[:-1],1)[0][0][:num_img].to(device)
                        c_l = random.sample(train_data[:-1],1)[0][0][:num_img].to(device)

                        out1 = model_d(c_g)[0]
                        out2 = model_d(c_g_previous)[0]
                        out3 = model_d(c_l)[0]


                        loss_c1 = (T ** 2) * criterion2(
                                torch.nn.functional.log_softmax(out1 / T, dim=1),
                                torch.nn.functional.softmax(out2 / T, dim=1)
                            )
                        # c2
                        loss_c2 = (T ** 2) * criterion2(
                                torch.nn.functional.log_softmax(out1 / T, dim=1),
                                torch.nn.functional.softmax(out3 / T, dim=1)
                            )
                        # c3
                        logit = hard_2_logit(lst=input_label,num_classes=args.total_class).to(device)
                        loss_c3 = (T ** 2) * criterion2(
                                torch.nn.functional.log_softmax(out1 / T, dim=1),
                                torch.nn.functional.softmax(logit / T, dim=1)
                            )

                        loss_all = loss1 + d_loss + loss_c1 + loss_c2 + loss_c3
                    else:
                        loss_all = loss1 + d_loss

                    loss_all.backward()

                    current_B+=1
                    d_optimizer.step()
                    if current_B >= num:
                        break

                    # 训练生成
                    input_label = Variable(torch.randint(0, id*args.task_class, (num_img,))).to(device)
                    noise = Variable(torch.randn(num_img, noise_dimension)).to(device)
                    fake_img = model_g(noise,input_label)
                    lob, output = model_d(fake_img)
                    output = output.flatten()
                    g_loss_dis = criterion(output, real_label)
                    g_loss_ce = criterion(lob, input_label)
                    g_loss = g_loss_ce + g_loss_dis
                    g_optimizer.zero_grad()
                    g_loss.backward()
                    
                    current_B+=1

                    g_optimizer.step()
                    g_batch_loss.append(g_loss.item())
                    if current_B >= num:
                        break

                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                if len(g_batch_loss) != 0:
                    g_epoch_loss.append(sum(g_batch_loss) / len(g_batch_loss))
                d_epoch_loss.append(sum(d_batch_loss) / len(d_batch_loss))
                scheduler.step(sum(batch_loss) / len(batch_loss))
                if (epoch == args.epochs -1) or (current_B >= num):
                    print('Client Index = {}\tEpoch: {}\timg_nums: {}\tEpoch_Loss: {:.10f}'.format(
                        self.id, epoch, sum(img_num_list),sum(batch_loss) / len(batch_loss)))
                if current_B >= num:
                    print('--------------------B has been used in '+str(epoch)+' epoch '+str(len(batch_loss))+' batch-----------')
                    break
            return current_B
        except Exception as e:
            logging.error(traceback.format_exc())

    

    