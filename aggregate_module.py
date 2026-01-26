import copy
import torch
import numpy as np
from collections import OrderedDict
from sklearn.cluster import KMeans,SpectralClustering
from scipy.cluster.hierarchy import dendrogram,linkage,fcluster
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score,calinski_harabasz_score
import heapq
import math


def aggregate_params_with_weight_labels(FL, weights):
    new_params = OrderedDict()
    # 获取模型参数的键并存储在keys列表中
    keys = FL['user' + str(0)]['model'].state_dict().keys()
    for key in keys:
        if key.startswith('conv1'):
            # if 'conv1' in key:
            new_params[key] = FL['user' + str(0)]['model'].state_dict()[key]
        elif key.startswith('linear'):
            new_params[key] = FL['user' + str(0)]['model'].state_dict()[key]
        else:# 遍历标签为i的索引
            # print(key)
            for j in range(len(weights)):
                if key not in new_params:
                    new_params[key] = FL['user' + str(j)]['model'].state_dict()[key] * weights[j]
                else:
                    new_params[key] = new_params[key] + FL['user' + str(j)]['model'].state_dict()[key] * weights[j]
    weights = weights.tolist()
    return new_params, weights


def new_acc_avg(FL, weights, acc, num_client):
#     计算b
    client_acc = acc.tolist()
    client_b = [0 for i in range(num_client)]
    for i in range(num_client):
        client_b[i] = client_acc[i]/(100 - client_acc[i])
    weight_b = np.array(client_b) / sum(np.array(client_b))
    weight_b = weight_b.tolist() 
    
#     计算a
    weight_aa = weights
    # 使用 heapq 模块获取acc列表中的前两个最大值
    largest_values = heapq.nlargest(2, client_acc)
    first_max_value = largest_values[0]
    second_max_value = largest_values[1]
    client_a = [0 for i in range(num_client)]
    for i in range(num_client):
        if client_acc[i] == first_max_value:
            client_a[i] = weight_aa[i]/(-np.log((first_max_value/100 - second_max_value/100)))
        else:
            client_a[i] = weight_aa[i]/(-np.log((first_max_value/100 - client_acc[i]/100)))
    # weight_a = np.array(client_a) / sum(np.array(client_a))
    # weight_a = weight_a.tolist() 
#     计算c
    client_c = [0 for i in range(num_client)]
    for i in range(num_client):
        client_c[i] = client_a[i] + weight_b[i]
    weight_c = np.array(client_c) / sum(np.array(client_c))
    
    new_params = OrderedDict()
    # 获取模型参数的键并存储在keys列表中
    keys = FL['user' + str(0)]['model'].state_dict().keys()
    for key in keys:
        if key.startswith('conv1'):
            new_params[key] = FL['user' + str(0)]['model'].state_dict()[key]
        elif key.startswith('linear'):
            new_params[key] = FL['user' + str(0)]['model'].state_dict()[key]
        else:# 遍历标签为i的索引
            for j in range(num_client):
                if key not in new_params:
                    new_params[key] = FL['user' + str(j)]['model'].state_dict()[key] * weight_c[j]
                else:
                    new_params[key] = new_params[key] + FL['user' + str(j)]['model'].state_dict()[key] * weight_c[j]
    weight_c = weight_c.tolist() 
    return new_params, weight_b, client_a, weight_c




def cluster_update(models,params):
  
    models['0'].load_state_dict(params)

    return models


def cluster_distribute(global_model,FL_dict,labels=None):
    if labels is None:
        for i in range(len(FL_dict)):
            model_dict = copy.deepcopy(global_model['0'].state_dict())
            mapping = OrderedDict((k, v) for k, v in FL_dict['user' + str(i)]['model'].state_dict().items() if k.startswith('conv1'))
            fc = OrderedDict((k, v) for k, v in FL_dict['user' + str(i)]['model'].state_dict().items() if k.startswith('linear'))
            for key, value in mapping.items():
                if key in model_dict:
                    model_dict[key] = value
            for key, value in fc.items():
                if key in model_dict:
                    model_dict[key] = value
            FL_dict['user' + str(i)]['model'].load_state_dict(model_dict)
    return FL_dict
    

def cluster_aggregate(mode,warm_up,epoch,FL_dict,weights, xs_weight, acc, num_client, select_client=None,cluster_num=None):
    # print("weight before cluster_aggregate: ", weights)
    # weights = np.array(weights) / sum(np.array(weights)) # 根据每个client上的数据量占所有clients数据总量的比例，作为权重
    if epoch > 0:
        weights = np.array(xs_weight[epoch])
    else:
        weights = np.array(weights) / sum(np.array(weights)) 
    print("weight in cluster_aggregate: ", weights)
    keys = FL_dict['user' + str(0)]['model'].state_dict().keys()
    cluster_key = list(keys)[-2]

 
    new_params, weights = aggregate_params_with_weight_labels(FL_dict, weights)   
    return new_params, weights
     

