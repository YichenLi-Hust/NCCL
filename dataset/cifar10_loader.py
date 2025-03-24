import json
import logging
import os
import random
import numpy as np
import torch
# import matplotlib.pyplot as plt
import h5py
import torch
from torchvision import datasets, transforms
torch.set_printoptions(sci_mode=False)

def _data_transforms_cifar10():
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])

    valid_transform = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])

    return train_transform, valid_transform

def fdil_batch_data(data, batch_size, model_name="resnet"):

    data_x = data['x']
    data_y = data['y']

    # if model_name != "lr":
    #     data_x = np.array(data_x).reshape(-1, 3, crop_size, crop_size)


    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    batch_data = list()
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i + batch_size]
        batched_y = data_y[i:i + batch_size]

        batched_x = [i.tolist() for i in batched_x]
        batched_y = [i.tolist() for i in batched_y]

        batched_x = torch.from_numpy(np.asarray(batched_x)).float()
        batched_y = torch.from_numpy(np.asarray(batched_y)).long()
        batch_data.append((batched_x, batched_y))
    return batch_data


def partition_data(label_list, n_nets, alpha):
    # logging.info("*********partition data***************")
    min_size = 0
    K = 10
    N = len(label_list)
    # logging.info("N = " + str(N))
    net_dataidx_map = {}

    while min_size < 10:
        idx_batch = [[] for _ in range(n_nets)]
        # for each class in the dataset
        for k in range(K):
            idx_k = np.where(label_list == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
            ## Balance
            proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_nets):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]

    return net_dataidx_map


def dirichlet_distribution(grouped_data, map, new_users, maximun, batch_size):
    X_train, Y_train, X_test, Y_test = grouped_data
    train_idx_map, test_idx_map = map

    new_train_data = {}
    new_test_data = {}

    for index, idx_list in train_idx_map.items():
        key = new_users[index]
        temp_data = {"x": [X_train[i] for i in idx_list],
                     "y": [Y_train[i] for i in idx_list]}
        new_train_data[key] = temp_data

    for index, idx_list in test_idx_map.items():
        key = new_users[index]
        temp_data = {"x": [X_test[i] for i in idx_list],
                     "y": [Y_test[i] for i in idx_list]}

        # print(temp_data["x"][0], temp_data["x"][0].size())
        # print(temp_data["y"][0])
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
            train_batch = fdil_batch_data(new_data, batch_size)
        else:
            train_data_local_num_dict[client_idx] = user_train_data_num
            train_batch = fdil_batch_data(new_train_data[u], batch_size)
        test_batch = fdil_batch_data(new_test_data[u], batch_size)

        train_data_local_dict[client_idx] = train_batch
            # test_data_local_dict[client_idx] = test_batch
        test_data_global += test_batch
        client_idx += 1
    random.shuffle(test_data_global)
    test_data_global = test_data_global[:len(test_data_global)//10]

    return train_data_local_num_dict, train_data_local_dict,test_data_global
    

def load_cifar10(client_num_in_total,alpha,batch_size,maximun):
    class_num = 10
    n = 2
    labels = range(class_num)
    new_users = []

    for i in range(client_num_in_total):
        if i < 10:
            new_users.append("f_0000" + str(i))
        else:
            new_users.append("f_000" + str(i))


    train_transform, test_transform = _data_transforms_cifar10()

    trainset = datasets.CIFAR10(root='/home/ycli/Dataset/CIFAR10', train=True,
                                            download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset),
                                            shuffle=False, num_workers=2)


    testset = datasets.CIFAR10(root='/home/ycli/Dataset/CIFAR10', train=False,
                                        download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset),
                                            shuffle=False, num_workers=2)


    X_train, Y_train = next(iter(trainloader))
    X_test, Y_test = next(iter(testloader))

    grouped_data = []
    for i in range(0, len(labels), n):
        if i+n > len(labels):  
            continue
        label_group = labels[i:i+n]
            
        X_train_group = X_train[torch.any(Y_train.unsqueeze(1) == torch.tensor(label_group), dim=1)]
        Y_train_group = Y_train[torch.any(Y_train.unsqueeze(1) == torch.tensor(label_group), dim=1)]
            
        X_test_group = X_test[torch.any(Y_test.unsqueeze(1) == torch.tensor(label_group), dim=1)]
        Y_test_group = Y_test[torch.any(Y_test.unsqueeze(1) == torch.tensor(label_group), dim=1)]
            
        grouped_data.append((X_train_group, Y_train_group, X_test_group, Y_test_group))

    logging.info('load init data')
    train_map = partition_data(grouped_data[0][1], client_num_in_total, alpha)
    test_map  = partition_data(grouped_data[0][3], client_num_in_total, alpha)
    map = (train_map, test_map)
    train_num_dict, init_train, init_test = dirichlet_distribution(grouped_data[0], map, new_users, maximun, batch_size)

    logging.info('load increment data1')
    train_map1 = partition_data(grouped_data[1][1], client_num_in_total, alpha)
    test_map1  = partition_data(grouped_data[1][3], client_num_in_total, alpha)
    map1 = (train_map1, test_map1)
    _, incre_train1, incre_test1 = dirichlet_distribution(grouped_data[1], map1, new_users, maximun, batch_size)   

    logging.info('load increment data2')
    train_map2 = partition_data(grouped_data[2][1], client_num_in_total, alpha)
    test_map2  = partition_data(grouped_data[2][3], client_num_in_total, alpha)
    map2 = (train_map2, test_map2)
    _, incre_train2, incre_test2 = dirichlet_distribution(grouped_data[2], map2, new_users, maximun, batch_size)   

    logging.info('load increment data3')
    train_map3 = partition_data(grouped_data[3][1], client_num_in_total, alpha)
    test_map3  = partition_data(grouped_data[3][3], client_num_in_total, alpha)
    map3 = (train_map3, test_map3)
    _, incre_train3, incre_test3 = dirichlet_distribution(grouped_data[3], map3, new_users, maximun, batch_size)   

    logging.info('load increment data4')
    train_map4 = partition_data(grouped_data[4][1], client_num_in_total, alpha)
    test_map4  = partition_data(grouped_data[4][3], client_num_in_total, alpha)
    map4 = (train_map4, test_map4)
    _, incre_train4, incre_test4 = dirichlet_distribution(grouped_data[4], map4, new_users, maximun, batch_size)   

    # logging.info('load increment data5')
    # train_map5 = partition_data(grouped_data[5][1], client_num_in_total, alpha)
    # test_map5  = partition_data(grouped_data[5][3], client_num_in_total, alpha)
    # map5 = (train_map5, test_map5)
    # _, incre_train5, incre_test5 = dirichlet_distribution(grouped_data[5], map5, new_users, maximun, batch_size)   

    # logging.info('load increment data6')
    # train_map6 = partition_data(grouped_data[6][1], client_num_in_total, alpha)
    # test_map6  = partition_data(grouped_data[6][3], client_num_in_total, alpha)
    # map6 = (train_map6, test_map6)
    # _, incre_train6, incre_test6 = dirichlet_distribution(grouped_data[6], map6, new_users, maximun, batch_size)   

    # logging.info('load increment data7')
    # train_map7 = partition_data(grouped_data[7][1], client_num_in_total, alpha)
    # test_map7  = partition_data(grouped_data[7][3], client_num_in_total, alpha)
    # map7 = (train_map7, test_map7)
    # _, incre_train7, incre_test7 = dirichlet_distribution(grouped_data[7], map7, new_users, maximun, batch_size)   

    # logging.info('load increment data8')
    # train_map8 = partition_data(grouped_data[8][1], client_num_in_total, alpha)
    # test_map8  = partition_data(grouped_data[8][3], client_num_in_total, alpha)
    # map8 = (train_map8, test_map8)
    # _, incre_train8, incre_test8 = dirichlet_distribution(grouped_data[8], map8, new_users, maximun, batch_size)   

    # logging.info('load increment data9')
    # train_map9 = partition_data(grouped_data[9][1], client_num_in_total, alpha)
    # test_map9  = partition_data(grouped_data[9][3], client_num_in_total, alpha)
    # map9 = (train_map9, test_map9)
    # _, incre_train9, incre_test9 = dirichlet_distribution(grouped_data[9], map9, new_users, maximun, batch_size)   


    incremental_train_data = {}
    incremental_test_data = {}
    for i in range(client_num_in_total):
        incremental_train_data[i] = []
        incremental_test_data[i] = []
       
    for i in range(client_num_in_total):
        incremental_train_data[i].append(incre_train1[i])
        incremental_train_data[i].append(incre_train2[i])
        incremental_train_data[i].append(incre_train3[i])
        incremental_train_data[i].append(incre_train4[i])
        # incremental_train_data[i].append(incre_train5[i])
        # incremental_train_data[i].append(incre_train6[i])
        # incremental_train_data[i].append(incre_train7[i])
        # incremental_train_data[i].append(incre_train8[i])
        # incremental_train_data[i].append(incre_train9[i])
        

        incremental_test_data[i].append(incre_test1)
        incremental_test_data[i].append(incre_test2)
        incremental_test_data[i].append(incre_test3)
        incremental_test_data[i].append(incre_test4)
        # incremental_test_data[i].append(incre_test5)
        # incremental_test_data[i].append(incre_test6)
        # incremental_test_data[i].append(incre_test7)
        # incremental_test_data[i].append(incre_test8)
        # incremental_test_data[i].append(incre_test9)
    
    # print(init_test)
    # print(incremental_test_data)

    return [train_num_dict, init_train, init_test, incremental_train_data, incremental_test_data, class_num]


if __name__ == '__main__':
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    print("finish")