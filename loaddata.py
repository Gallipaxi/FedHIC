import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from scipy.stats import dirichlet
import random
from scipy.ndimage import uniform_filter1d
from collections import OrderedDict
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import os
from PIL import Image
import pandas as pd
import scipy.io as scio
from functools import reduce
from sklearn.model_selection import train_test_split
import math
from torch.utils.data import DataLoader,random_split,Dataset



def val_tinyimagenet(val_path, val_annotations_path):
    # 从注释文件中读取图像名和对应的标签
    df = pd.read_csv(val_annotations_path, sep='\t', header=None, names=['File', 'Class', 'X', 'Y', 'H', 'W'])
    file_to_class = {row[1]['File']: row[1]['Class'] for row in df.iterrows()}

    data, labels = [], []
    for file_name in os.listdir(val_path):
        # 打开图像
        img = Image.open(os.path.join(val_path, file_name))
        img_array = np.array(img)
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        # 获取图像的类别
        label = file_to_class[file_name]
        data.append(img_array)
        labels.append(label)
    testset=[np.array(data),labels]
    return testset

def train_tinyimagenet(path,wnids_path):
    with open(wnids_path,'r') as f:
        wnids=[x.strip() for x in f]
    traindata,trainlabel=[],[]
    for i, wnid in enumerate(wnids):
        for img_path in os.listdir(os.path.join(path,wnid,'images')):
            img=Image.open(os.path.join(path,wnid,'images',img_path))
            img=np.array(img)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, axis=-1)
            traindata.append(img)
            trainlabel.append(wnid)
    trainset=[np.array(traindata),trainlabel]
    return trainset


def load_dataset(name):
    dir='F:/FedHIC'
    input = scio.loadmat(os.path.join(dir, name) + '/' + name + '.mat')
    gt = scio.loadmat(os.path.join(dir, name) + '/' + name + '_gt.mat')

    # 读取mat中的数据和gt
    gt = gt[name + '_gt']
    # input = input[name]
    input = np.float32(input[name])

    # 数据归一化
    for i in range(input.shape[-1]):
        input[:, :, i] = (input[:, :, i] - np.mean(input[:, :, i])) / np.std(input[:, :, i])

    # 统计每类样本数量
    dict_k = {}
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            if gt[i][j] != 0:
                if gt[i][j] not in dict_k:
                    dict_k[gt[i][j]] = 0
                dict_k[gt[i][j]] += 1
    dict_k = OrderedDict(sorted(dict_k.items(), key=lambda t: t[0]))
    total_num = reduce(lambda x, y: x + y, dict_k.values())  # 有label的总数量，后续作为数据的batch维度

    # 以非0-label像元为中心，截取patch并保存
    patch_size = 9  # 以像元为中心的patch的大小
    padding_size = patch_size // 2
    input = np.pad(input, ((padding_size, padding_size), (padding_size, padding_size), (0, 0)), 'constant')
    gt = np.pad(gt, padding_size, 'constant')

    data = np.zeros([total_num, patch_size, patch_size, input.shape[-1]])
    count_flag = 0  # 用于遍历data中total_num维度的计数器
    label_2_data = []
    for i in range(padding_size, gt.shape[0]-padding_size):
        for j in range(padding_size, gt.shape[1]-padding_size):
            if gt[i][j] != 0:
                label_2_data.append(gt[i][j])
                data[count_flag, :, :, :] = input[i - (patch_size // 2):i + (patch_size // 2) + 1,
                                            j - (patch_size // 2):j + (patch_size // 2) + 1, :]
                count_flag += 1
    data = data.transpose(0, 3, 1, 2)
    data_tensor = torch.from_numpy(data)
    label_2_data_tensor = torch.tensor(label_2_data)


    dataset_train, dataset_test, target_train, target_test = train_test_split(data_tensor, label_2_data_tensor,
                                                                              test_size=0.95, random_state=6)

    num_class = np.max(gt)
    num_per_class = []
    for i in range(num_class):
        num_per_class.append(dict_k[i + 1])

    trainset = [dataset_train, np.array(target_train)-1]
    testset = [dataset_test, np.array(target_test)-1]

    return trainset,testset,np.array(num_per_class)


class randompicker:
    def __init__(self,arr):
        self.arr=arr
        self.available=arr.copy()
    def pick(self):
        if not self.available:
            self.available=self.arr.copy()
        chosen=random.choice(self.available) # 从输入里面随机返回一个
        self.available.remove(chosen)
        return chosen


def produce_cluster_by_class(num_per_class,num_class_per_cluster):
    '''

    :param num_per_class: 每个类别的数量
    :param num_class_per_cluster: 每个聚类里面的类别数量
    :return: 返回根据类别计算出的聚类
    '''
    num_class=len(num_per_class) # 计算具体类别数量
    if sum(num_class_per_cluster) < num_class or any(i>num_class for i in num_class_per_cluster):
        raise Exception('The number of classes in clusters is not appropriate.')

    picker=randompicker(np.arange(num_class).tolist()) # 按照类别数量从0-n重新标，类别里面int可能不连续。然后从里面随机取出来一个
    cluster_by_class=[]
    print(num_class_per_cluster)
    for i in num_class_per_cluster: # num_class_per_cluster是list，i取里面的元素
        classes = []
        for j in range(i): # i表示i个类别，假设global一共有10个类别，i=2，j代表随机从10个类别里面不重复的选出2个类别
            classes.append(picker.pick())
        cluster_by_class.append(classes) # cluster_by_class里是列表嵌列表，子列表的数量是num_class_per_cluster中的元素数量，每个子列表里面存放随机选出来的类别
    return cluster_by_class

def produce_client_by_class(ran,num_per_class,num_client,cluster_by_class,num_class_per_cluster):
    '''

    :param ran:
    :param num_per_class: global类别中每个类别的数量
    :param num_client: clients的数量
    :param cluster_by_class:
    :param num_class_per_cluster: 每个聚类里面的类别数量
    :return:
    '''
    if ran:
        random_numbers=uniform_filter1d(np.random.rand(len(num_class_per_cluster)),size=3)
        num_client_per_cluster=np.round(random_numbers*num_client/np.sum(random_numbers)).astype(int)
        num_client_per_cluster[num_client_per_cluster<2]=2
        diff=np.sum(num_client_per_cluster)-num_client
        if diff>0:
            while diff!=0:
                ind=np.argmax(num_client_per_cluster)
                num_client_per_cluster[ind]-=1
                diff=diff-1
        elif diff<0:
            while diff!=0:
                 ind=np.argmin(num_client_per_cluster)
                 num_client_per_cluster[ind]+=1
                 diff=diff+1
        assert np.sum(num_client_per_cluster)==num_client
    client_by_class=[elem for i,elem in enumerate(cluster_by_class) for _ in range(num_client_per_cluster[i])]
    class_by_client=[]
    for i in range(len(num_per_class)):
        arr=[]
        for j,elem in enumerate(client_by_class):
            if i in elem:
                arr.append(j)
        class_by_client.append(arr)


    return client_by_class,class_by_client


def get_sample(clients,dirichlet_dis,is_train,dataset,classid,client_by_class,class_by_client):
    ind=np.random.permutation(len(dataset[1][dataset[1]==classid]))
    cur_data=dataset[0][dataset[1]==classid][ind]
    cur_label=dataset[1][dataset[1]==classid][ind]
    prob=dirichlet([dirichlet_dis]*len(class_by_client[classid]),seed=classid).rvs()
    # print(prob)
    class_num_by_client=np.round(len(ind)*prob)[0]
    diff=np.sum(class_num_by_client)-len(ind)
    if diff>0:
        while diff!=0:
            class_num_by_client[np.argmax(class_num_by_client)]-=1
            diff-=1
    elif diff<0:
        while diff!=0:
            class_num_by_client[np.argmin(class_num_by_client)]+=1
            diff+=1
    assert np.sum(class_num_by_client)==len(ind)
    while np.any(class_num_by_client<1):
        # breakpoint()
        class_num_by_client[np.argmax(class_num_by_client)] -= 1
        class_num_by_client[np.argmin(class_num_by_client)] += 1
    cur_cumsum=np.concatenate(([0],np.cumsum(class_num_by_client))).astype(int)
    for j in range(len(class_by_client[classid])):
        data_dict=[cur_data[cur_cumsum[j]:cur_cumsum[j+1]],cur_label[cur_cumsum[j]:cur_cumsum[j+1]]]
        if 'client'+str(class_by_client[classid][j]) not in clients:
            clients['client' + str(class_by_client[classid][j])] = {}
            clients['client' + str(class_by_client[classid][j])]['train']={}
            clients['client' + str(class_by_client[classid][j])]['test'] = {}
        if is_train:
            if clients['client' + str(class_by_client[classid][j])]['train']=={}:
                clients['client' + str(class_by_client[classid][j])]['train'] = data_dict
            else:
                clients['client'+str(class_by_client[classid][j])]['train'][0]=np.concatenate([clients['client'+str(class_by_client[classid][j])]['train'][0],data_dict[0]],axis=0)
                clients['client'+str(class_by_client[classid][j])]['train'][1]=np.concatenate([clients['client'+str(class_by_client[classid][j])]['train'][1],data_dict[1]])
        else:
            if clients['client' + str(class_by_client[classid][j])]['test'] == {}:
                clients['client' + str(class_by_client[classid][j])]['test'] = data_dict
            else:
                clients['client' + str(class_by_client[classid][j])]['test'][0] = np.concatenate(
                    [clients['client' + str(class_by_client[classid][j])]['test'][0], data_dict[0]], axis=0)
                clients['client' + str(class_by_client[classid][j])]['test'][1] = np.concatenate(
                    [clients['client' + str(class_by_client[classid][j])]['test'][1], data_dict[1]])

    return clients

def produce_client_by_data(trainset,testset,num_per_class,client_by_class,class_by_client,dirichlet_dis):
    num_class=len(num_per_class)
    clients={}
    for i in range(num_class):
        clients=get_sample(clients,dirichlet_dis,True,trainset,i,client_by_class,class_by_client)
        clients=get_sample(clients,dirichlet_dis,False,testset,i,client_by_class,class_by_client)
    clients=OrderedDict(clients)
    for i in range(len(clients)):
        value=clients.pop('client'+str(i))
        clients['client'+str(i)]=value
    return clients



def generate_dataloader(name,data,batch_size,transform=None):

    class Mydataset(Dataset):
        def __int__(self, data, label, transform=None):
            self.data = data
            self.label = label
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            sample_data = self.data[idx]
            if self.transform:
                sample_data = self.transform(sample_data)

            return sample_data, self.label[idx]

    dataset=Mydataset()
    dataset.data=data[0]
    dataset.label=data[1]
    dataset.transform=transform
    dataloader=DataLoader(dataset, batch_size=batch_size,shuffle=True,drop_last=True, worker_init_fn=lambda _: torch.manual_seed(42))

    return dataloader


def img_crop(input, box):
    '''

    :param img:
    :param box:
    :return: 按照box的维度对img进行切片并返回
    '''
    if len(input.shape) > 2:
        slice_data = input[box[0]:box[2], box[1]:box[3], :]
    else:
        slice_data = input[box[0]:box[2], box[1]:box[3]]
    return slice_data


def cal_num_of_class(gt): # 统计每类样本数量
    dict_k = {}
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            if gt[i][j] != 0:
                if gt[i][j] not in dict_k:
                    dict_k[gt[i][j]] = 0
                dict_k[gt[i][j]] += 1
    dict_k = OrderedDict(sorted(dict_k.items(), key=lambda t: t[0]))
    total_num = reduce(lambda x, y: x + y, dict_k.values())  # 有label的总数量，后续作为数据的batch维度
    return total_num


def data_split_in_client(img_slice, gt_slice, TR_gt_slice, TE_gt_slice, test_size, split, index_i):
    # 设置图像块和填充大小
    patch_size = 9
    padding_size = patch_size // 2

    # 对输入图像和标签进行填充
    input = np.pad(img_slice, ((padding_size, padding_size), (padding_size, padding_size), (0, 0)), 'constant')

    
    TR_gt = np.pad(TR_gt_slice, padding_size, 'constant')
    TE_gt = np.pad(TE_gt_slice, padding_size, 'constant')

    num_TR_gt = cal_num_of_class(TR_gt)
    num_TE_gt = cal_num_of_class(TE_gt)
    total_num = num_TR_gt + num_TE_gt

    data_TR = np.zeros([num_TR_gt, patch_size, patch_size, input.shape[-1]])
    data_TE = np.zeros([num_TE_gt, patch_size, patch_size, input.shape[-1]])

    gt = [TR_gt, TE_gt]

    TR_label_2_data = []
    TE_label_2_data = []

    for k in range(len(gt)):
        count_flag = 0
        for i in range(padding_size, TR_gt.shape[0] - padding_size):
            for j in range(padding_size, TR_gt.shape[1] - padding_size):
                if gt[k][i][j] != 0:
                    if k == 0:
                        # 将标签添加到label_2_data列表中
                        TR_label_2_data.append(gt[k][i][j])
                        # 在数据数组中存储相应的图像块
                        data_TR[count_flag, :, :, :] = input[i - (patch_size // 2):i + (patch_size // 2) + 1,
                                                        j - (patch_size // 2):j + (patch_size // 2) + 1, :]
                        count_flag += 1
                    else:
                        TE_label_2_data.append(gt[k][i][j])
                        data_TE[count_flag, :, :, :] = input[i - (patch_size // 2):i + (patch_size // 2) + 1,
                                                        j - (patch_size // 2):j + (patch_size // 2) + 1, :]
                        count_flag += 1

    data_TR = data_TR.transpose((0, 3, 1, 2))
    data_TE = data_TE.transpose((0, 3, 1, 2))

    trainset = [data_TR, np.array(TR_label_2_data) - 1]
    testset = [data_TE, np.array(TE_label_2_data) - 1]

    # if index_i in [0, 3]:
    #     print('走了这边')
    #     TR_gt = np.pad(TR_gt_slice, padding_size, 'constant')
    #     TE_gt = np.pad(TE_gt_slice, padding_size, 'constant')

    #     num_TR_gt = cal_num_of_class(TR_gt)
    #     num_TE_gt = cal_num_of_class(TE_gt)
    #     total_num = num_TR_gt + num_TE_gt

    #     data_TR = np.zeros([num_TR_gt, patch_size, patch_size, input.shape[-1]])
    #     data_TE = np.zeros([num_TE_gt, patch_size, patch_size, input.shape[-1]])

    #     gt = [TR_gt, TE_gt]

    #     TR_label_2_data = []
    #     TE_label_2_data = []

    #     for k in range(len(gt)):
    #         count_flag = 0
    #         for i in range(padding_size, TR_gt.shape[0] - padding_size):
    #             for j in range(padding_size, TR_gt.shape[1] - padding_size):
    #                 if gt[k][i][j] != 0:
    #                     if k == 0:
    #                         # 将标签添加到label_2_data列表中
    #                         TR_label_2_data.append(gt[k][i][j])
    #                         # 在数据数组中存储相应的图像块
    #                         data_TR[count_flag, :, :, :] = input[i - (patch_size // 2):i + (patch_size // 2) + 1,
    #                                                        j - (patch_size // 2):j + (patch_size // 2) + 1, :]
    #                         count_flag += 1
    #                     else:
    #                         TE_label_2_data.append(gt[k][i][j])
    #                         data_TE[count_flag, :, :, :] = input[i - (patch_size // 2):i + (patch_size // 2) + 1,
    #                                                        j - (patch_size // 2):j + (patch_size // 2) + 1, :]
    #                         count_flag += 1

    #     data_TR = data_TR.transpose((0, 3, 1, 2))
    #     data_TE = data_TE.transpose((0, 3, 1, 2))

    #     trainset = [data_TR, np.array(TR_label_2_data) - 1]
    #     testset = [data_TE, np.array(TE_label_2_data) - 1]


    # else:
    #     gt = np.pad(gt_slice, padding_size, 'constant')
    #     # 计算标签中类别的总数
    #     total_num = cal_num_of_class(gt_slice)
    #     # 创建用于存储数据的数组
    #     data = np.zeros([total_num, patch_size, patch_size, input.shape[-1]])
    #     # 用于跟踪索引和标签映射的变量
    #     count_flag = 0
    #     label_2_data = []

    #     # 遍历填充后的标签数组
    #     for i in range(padding_size, gt.shape[0] - padding_size):
    #         for j in range(padding_size, gt.shape[1] - padding_size):
    #             if gt[i][j] != 0:
    #                 # 将标签添加到label_2_data列表中
    #                 label_2_data.append(gt[i][j])
    #                 # 在数据数组中存储相应的图像块
    #                 data[count_flag, :, :, :] = input[i - (patch_size // 2):i + (patch_size // 2) + 1,
    #                                             j - (patch_size // 2):j + (patch_size // 2) + 1, :]
    #                 count_flag += 1

    #     # 转置数据数组以匹配预期的格式
    #     data = data.transpose(0, 3, 1, 2)

    #     # 将数据和标签拆分为训练集和测试集
    #     dataset_train, dataset_test, target_train, target_test = train_test_split(data, label_2_data,
    #                                                                               test_size=test_size, random_state=6)
    #     # 创建 trainset和 testset列表
    #     trainset = [dataset_train, np.array(target_train) - 1]
    #     testset = [dataset_test, np.array(target_test) - 1]


    return trainset, testset



def preprocess(img, gt, TR_gt, TE_gt, test_size, split, index_i):
    # 领取标签和特征
    img = np.float32(img)
    num_class = np.max(TR_gt)

    # 数据归一化
  
    for i in range(img.shape[-1]):
        img[:, :, i] = (img[:, :, i] - np.mean(img[:, :, i])) / np.std(img[:, :, i])

    # 将整个图片进行训练集和测试集的划分
    client_train, testset = data_split_in_client(img, gt, TR_gt, TE_gt, test_size, split, index_i)


    return client_train, testset, num_class
