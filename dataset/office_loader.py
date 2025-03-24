import json
import logging
import os
import math
import random
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

def non_iid_partition_with_dirichlet_distribution(label_list,
                                                  client_num,
                                                  classes,
                                                  alpha,
                                                  task='classification'):
   
    net_dataidx_map = {}
    K = classes
    N = len(label_list)

    min_size = 0
    while min_size < 10:
        idx_batch = [[] for _ in range(client_num)]

        for k in range(K):
            idx_k = np.where(label_list == k)[0]
            idx_batch, min_size = partition_class_samples_with_dirichlet_distribution(N, alpha, client_num,
                                                                                      idx_batch, idx_k)
    for i in range(client_num):
        np.random.shuffle(idx_batch[i])
        net_dataidx_map[i] = idx_batch[i]

    return net_dataidx_map

def partition_class_samples_with_dirichlet_distribution(N, alpha, client_num, idx_batch, idx_k):
    np.random.shuffle(idx_k)
    proportions = np.random.dirichlet(np.repeat(alpha, client_num))

    proportions = np.array([p * (len(idx_j) < N / client_num) for p, idx_j in zip(proportions, idx_batch)])
    proportions = proportions / proportions.sum()
    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
    min_size = min([len(idx_j) for idx_j in idx_batch])

    return idx_batch, min_size


def load_office_home(client_num_in_total,alpha,batch_size):
    property = 0.8
    img_size = 256
    crop_size = 224
    class_num = 65
    maximun = 50000 # Memory size
    new_users = []
    root_path = "/Dataset/OfficeHome/"

    for i in range(client_num_in_total):
        if i < 10:
            new_users.append("f_0000" + str(i))
        else:
            new_users.append("f_000" + str(i))

    def fdil_batch_data(data, batch_size, model_name="resnet"):

        data_x = data['x']
        data_y = data['y']

        if model_name != "lr":
            data_x = np.array(data_x).reshape(-1, 3, crop_size, crop_size)

        data_domain = data['domain']
        data_outer_score = [0.5]*len(data_x)  # across domain
        data_inner_score = [0]*len(data_x)     # in a domain

        # randomly shuffle data
        np.random.seed(100)
        rng_state = np.random.get_state()
        np.random.shuffle(data_x)
        np.random.set_state(rng_state)
        np.random.shuffle(data_y)
        # data_x = np.where(data_x > 0, 1, 0)
        # loop through mini-batches
        batch_data = list()
        for i in range(0, len(data_x), batch_size):
            batched_x = data_x[i:i + batch_size]
            batched_y = data_y[i:i + batch_size]
            batched_domain = data_domain[i:i + batch_size]
            batched_inner_score = data_inner_score[i:i + batch_size]
            batched_outer_score = data_outer_score[i:i + batch_size]
            batched_x = torch.from_numpy(np.asarray(batched_x)).float()
            batched_y = torch.from_numpy(np.asarray(batched_y)).long()
            batch_data.append((batched_x, batched_y, batched_domain, batched_inner_score, batched_outer_score))
        return batch_data

    transform_office = transforms.Compose([
        transforms.Resize([img_size,img_size]),
        # transforms.Grayscale(num_output_channels=1),
        transforms.RandomCrop(crop_size),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(0.5,0.5)
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    def dirichlet_distribution(all_train_data,all_test_data):
        new_train_data = {}
        new_test_data = {}

        train_label_list = np.asarray(all_train_data["y"])
        train_idx_map = non_iid_partition_with_dirichlet_distribution(train_label_list, client_num_in_total, class_num,
                                                                     alpha)

        for index, idx_list in train_idx_map.items():
            key = new_users[index]
            temp_data = {"x": [all_train_data["x"][i] for i in idx_list],
                         "y": [all_train_data["y"][i] for i in idx_list],
                         "domain": [all_train_data["domain"][i] for i in idx_list]}
            new_train_data[key] = temp_data

        test_label_list = np.asarray(all_test_data["y"])
        test_idx_map = non_iid_partition_with_dirichlet_distribution(test_label_list, client_num_in_total, class_num,
                                                                     alpha)

        for index, idx_list in test_idx_map.items():
            key = new_users[index]
            temp_data = {"x": [all_test_data["x"][i] for i in idx_list],
                         "y": [all_test_data["y"][i] for i in idx_list],
                         "domain": [all_test_data["domain"][i] for i in idx_list]}
            new_test_data[key] = temp_data

        train_data_num = 0
        test_data_num = 0
        train_data_local_dict = dict()
        test_data_local_dict = dict()
        train_data_local_num_dict = dict()
        test_data_global = list()
        client_idx = 0
        for u in new_users:
            user_train_data_num = len(new_train_data[u]['x'])
            user_test_data_num = len(new_test_data[u]['x'])
            train_data_num += user_train_data_num
            test_data_num += user_test_data_num
            if user_train_data_num > maximun:
                train_data_local_num_dict[client_idx] = maximun
                new_data = {}
                new_data["x"] = new_train_data[u]["x"][:maximun]
                new_data["y"] = new_train_data[u]["y"][:maximun]
                new_data["domain"] = new_train_data[u]["domain"][:maximun]
                train_batch = fdil_batch_data(new_data, batch_size)
            else:
                train_data_local_num_dict[client_idx] = user_train_data_num
                train_batch = fdil_batch_data(new_train_data[u], batch_size)
            test_batch = fdil_batch_data(new_test_data[u], batch_size)

            train_data_local_dict[client_idx] = train_batch
            # test_data_local_dict[client_idx] = test_batch
            test_data_global += test_batch
            client_idx += 1

        return train_data_local_num_dict, train_data_local_dict,test_data_global
    
    DOMAIN = ["Art","Clipart","Product","Real World"]

    def get_init_data(index):
        temp = {"x":[],"y":[],"domain":[]}
        all_train_data = {"x":[],"y":[],"domain":[]}
        all_test_data = {"x":[],"y":[],"domain":[]}
        this_path = root_path + DOMAIN[index]
        class_item = os.listdir(this_path)

        for (y_idx,i) in enumerate(class_item):
            cur_path = this_path + '/' + i
            all = len(os.listdir(cur_path))
            # train_num = math.ceil(all*property)

            count = 0
            for j in os.listdir(cur_path):
                pil_photo = Image.open(cur_path+'/'+j).convert('RGB')
                photo = transform_office(pil_photo).flatten(0).tolist()

                # if count < train_num:
                temp["x"].append(photo)
                temp["y"].append(y_idx)
                temp["domain"].append(y_idx)
                # else:
                #     all_test_data["x"].append(photo)
                #     all_test_data["y"].append(y_idx)
                #     all_test_data["domain"].append(y_idx)
                # count += 1
        all = len(temp['x'])
        data_x = temp['x']
        data_y = temp["y"]
        data_domain = temp["domain"]
        rng_state = np.random.get_state()
        np.random.shuffle(data_x)
        np.random.set_state(rng_state)
        np.random.shuffle(data_y)
        np.random.set_state(rng_state)
        np.random.shuffle(data_domain)

        lst = np.random.choice(range(all), math.ceil(all*property), replace=False)
        for i in range(all):
            if i in lst:
                all_train_data["x"].append(temp["x"][i])
                all_train_data["y"].append(temp["y"][i])
                all_train_data["domain"].append(temp["domain"][i])
            else:
                all_test_data["x"].append(temp["x"][i])
                all_test_data["y"].append(temp["y"][i])
                all_test_data["domain"].append(temp["domain"][i])


        return dirichlet_distribution(all_train_data, all_test_data)
    
    def get_incre1_data(index):
        all_train_data = {"x":[],"y":[],"domain":[]}
        all_test_data = {"x":[],"y":[],"domain":[]}
        this_path = root_path + DOMAIN[index]
        class_item = os.listdir(this_path)

        for (y_idx,i) in enumerate(class_item):
            cur_path = this_path + '/' + i
            all = len(os.listdir(cur_path))
            train_num = math.ceil(all*property)

            count = 0
            for j in os.listdir(cur_path):
                pil_photo = Image.open(cur_path+'/'+j).convert("RGB")
                photo = transform_office(pil_photo).flatten(0).tolist()

                if count < train_num:
                    all_train_data["x"].append(photo)
                    all_train_data["y"].append(y_idx)
                    all_train_data["domain"].append(y_idx+65)
                else:
                    all_test_data["x"].append(photo)
                    all_test_data["y"].append(y_idx)
                    all_test_data["domain"].append(y_idx+65)
                count += 1
        return dirichlet_distribution(all_train_data, all_test_data)
    
    def get_incre2_data(index):
        all_train_data = {"x":[],"y":[],"domain":[]}
        all_test_data = {"x":[],"y":[],"domain":[]}
        this_path = root_path + DOMAIN[index]
        class_item = os.listdir(this_path)

        for (y_idx,i) in enumerate(class_item):
            cur_path = this_path + '/' + i
            all = len(os.listdir(cur_path))
            train_num = math.ceil(all*property)

            count = 0
            for j in os.listdir(cur_path):
                pil_photo = Image.open(cur_path+'/'+j).convert('RGB')
                photo = transform_office(pil_photo).flatten(0).tolist()

                if count < train_num:
                    all_train_data["x"].append(photo)
                    all_train_data["y"].append(y_idx)
                    all_train_data["domain"].append(y_idx+130)
                else:
                    all_test_data["x"].append(photo)
                    all_test_data["y"].append(y_idx)
                    all_test_data["domain"].append(y_idx+130)
                count += 1
        return dirichlet_distribution(all_train_data, all_test_data)

    def get_incre3_data(index):
        all_train_data = {"x":[],"y":[],"domain":[]}
        all_test_data = {"x":[],"y":[],"domain":[]}
        this_path = root_path + DOMAIN[index]
        class_item = os.listdir(this_path)

        for (y_idx,i) in enumerate(class_item):
            cur_path = this_path + '/' + i
            all = len(os.listdir(cur_path))
            train_num = math.ceil(all*property)

            count = 0
            for j in os.listdir(cur_path):
                pil_photo = Image.open(cur_path+'/'+j).convert('RGB')
                photo = transform_office(pil_photo).flatten(0).tolist()

                if count < train_num:
                    all_train_data["x"].append(photo)
                    all_train_data["y"].append(y_idx)
                    all_train_data["domain"].append(y_idx+195)
                else:
                    all_test_data["x"].append(photo)
                    all_test_data["y"].append(y_idx)
                    all_test_data["domain"].append(y_idx+195)
                
                count += 1
        return dirichlet_distribution(all_train_data, all_test_data)

    logging.info('load init data')
    train_num_dict, init_train, init_test = get_init_data(0)
    # print(init_train[0][0][0][0])

    logging.info('load increment data1')
    # _, incre_train1, incre_test1 = get_incre1_data(1)
    logging.info('load increment data2')
    # _, incre_train2, incre_test2 = get_incre2_data(2)
    logging.info('load increment data3')
    # _, incre_train3, incre_test3 = get_incre3_data(3)


    incremental_train_data = {}
    incremental_test_data = {}
    for i in range(client_num_in_total):
        incremental_train_data[i] = []
        incremental_test_data[i] = []
       
    # for i in range(client_num_in_total):
    #     incremental_train_data[i].append(incre_train1[i])
    #     incremental_train_data[i].append(incre_train2[i])
    #     incremental_train_data[i].append(incre_train3[i])

    #     incremental_test_data[i].append(incre_test1)
    #     incremental_test_data[i].append(incre_test2)
    #     incremental_test_data[i].append(incre_test3)

    return [train_num_dict, init_train, init_test, incremental_train_data, incremental_test_data, class_num]


if __name__ == '__main__':
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    load_office_home(20,0.1,32)
    print("finish")