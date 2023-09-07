
from tqdm import tqdm
#from model import BotGAT
import json
import torch
import torch.nn as nn
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import accuracy_score, f1_score
import random

import torch
import os.path as osp
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, \
    roc_auc_score, precision_recall_curve, auc
import torch
import torch.nn.functional as func


def get_transfer_data():
    path = '../../BotRGCN/twibot_22/processed_data'
    labels = torch.load(osp.join(path, 'label.pt'))
    des_embedding = torch.load(osp.join(path, 'des_tensor.pt'))
    tweet_embedding = torch.load(osp.join(path, 'tweets_tensor.pt'))
    num_property_embedding = torch.load(osp.join(path, 'num_properties_tensor.pt'))
    cat_property_embedding = torch.load(osp.join(path, 'cat_properties_tensor.pt'))
    edge_index = torch.load(osp.join(path, 'edge_index.pt'))
    num_for_h=torch.load('data/Twibot-20/num_properties_tensor.pt')
    return Data(edge_index=edge_index,
                y=labels,
                des_embedding=des_embedding,
                tweet_embedding=tweet_embedding,
                num_property_embedding=num_property_embedding,
                cat_property_embedding=cat_property_embedding,
                num_nodes=labels.shape[0],
                num_for_h=num_for_h)


data_index = {
    'cresci-2015': 'cresci_15',
    'Twibot-22': 'twibot_22',
    'Twibot-20': 'twibot_20'
}


def get_train_data(dataset_name):
    path = '/data3/whr/lyh/MoE/mixture-of-experts/BotRGCN/{}/processed_data'.format(data_index[dataset_name])
    if not osp.exists(path):
        raise KeyError
    labels = torch.load(osp.join(path, 'label.pt'))
    des_embedding = torch.load(osp.join(path, 'des_tensor.pt'))
    tweet_embedding = torch.load(osp.join(path, 'tweets_tensor.pt'))
    num_property_embedding = torch.load(osp.join(path, 'num_properties_tensor.pt'))
    cat_property_embedding = torch.load(osp.join(path, 'cat_properties_tensor.pt'))
    edge_type = torch.load(osp.join(path, 'edge_type.pt'))
    edge_index = torch.load(osp.join(path, 'edge_index.pt'))
    num_for_h=torch.load('data/Twibot-20/num_properties_tensor.pt')
    num_for_h=torch.cat((torch.zeros((229580-11826,num_for_h.shape[-1])),num_for_h),dim=0)
    if dataset_name == 'Twibot-20':
        labels = torch.cat([labels, torch.empty(217754, dtype=torch.long).fill_(2)])
        train_idx = torch.arange(0, 8278)
        val_idx = torch.arange(8278, 8278 + 2365)
        test_idx = torch.arange(8278+2365, 8278 + 2365 + 1183)
    else:
        train_idx = torch.load(osp.join(path, 'train_idx.pt'))
        val_idx = torch.load(osp.join(path, 'val_idx.pt'))
        test_idx = torch.load(osp.join(path, 'test_idx.pt'))
    return Data(edge_index=edge_index,
                edge_type=edge_type,
                y=labels,
                num_for_h=num_for_h,
                des_embedding=des_embedding,
                tweet_embedding=tweet_embedding,
                num_property_embedding=num_property_embedding,
                cat_property_embedding=cat_property_embedding,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                num_nodes=des_embedding.shape[0])


def get_train_data_with_t5(dataset_name):
    path = '/data3/whr/lyh/MoE/mixture-of-experts/BotRGCN/{}/processed_data'.format(data_index[dataset_name])
    path_t5='data/T5/Twibot-20'
    if not osp.exists(path):
        raise KeyError
    labels = torch.load(osp.join(path, 'label.pt'))
    des_embedding = torch.load(osp.join(path, 'des_tensor.pt'))
    tweet_embedding = torch.load(osp.join(path, 'tweets_tensor.pt'))
    num_property_embedding = torch.load(osp.join(path, 'num_properties_tensor.pt'))
    cat_property_embedding = torch.load(osp.join(path, 'cat_properties_tensor.pt'))
    edge_type = torch.load(osp.join(path, 'edge_type.pt'))
    edge_index = torch.load(osp.join(path, 'edge_index.pt'))
    num_for_h=torch.load('data/Twibot-20/num_properties_tensor.pt')
    num_for_h=torch.cat((torch.zeros((229580-11826,num_for_h.shape[-1])),num_for_h),dim=0)
    des_t5=torch.load(path_t5+'/des_tensor.pt')
    tweets_t5=torch.load(path_t5+'/tweets_tensor.pt')
    if dataset_name == 'Twibot-20':
        labels = torch.cat([labels, torch.empty(217754, dtype=torch.long).fill_(2)])
        train_idx = torch.arange(0, 8278)
        val_idx = torch.arange(8278, 8278 + 2365)
        test_idx = torch.arange(8278+2365, 8278 + 2365 + 1183)
    else:
        train_idx = torch.load(osp.join(path, 'train_idx.pt'))
        val_idx = torch.load(osp.join(path, 'val_idx.pt'))
        test_idx = torch.load(osp.join(path, 'test_idx.pt'))
    return Data(edge_index=edge_index,
                edge_type=edge_type,
                y=labels,
                num_for_h=num_for_h,
                des_embedding=des_embedding,
                tweet_embedding=tweet_embedding,
                num_property_embedding=num_property_embedding,
                cat_property_embedding=cat_property_embedding,
                des_t5=des_t5,
                tweets_t5=tweets_t5,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                num_nodes=des_embedding.shape[0])




def get_new_train_data(dataset_name):
    path = '/data3/whr/lyh/MoE/mixture-of-experts/BotRGCN/{}/processed_data'.format(data_index[dataset_name])
    if not osp.exists(path):
        raise KeyError
    labels = torch.load(osp.join(path, 'label.pt'))
    des_embedding = torch.load(osp.join(path, 'des_tensor.pt'))
    tweet_embedding = torch.load(osp.join(path, 'tweets_tensor.pt'))
    num_property_embedding = torch.load(osp.join(path, 'normalized_new_num_fea.pt'))
    num_property_embedding=torch.cat((num_property_embedding,torch.zeros((229580-11826),27)),dim=0)
    cat_property_embedding = torch.load(osp.join(path, 'normalized_new_cat_fea.pt'))
    cat_property_embedding=torch.cat((cat_property_embedding,torch.zeros((229580-11826),11)),dim=0)
    
    edge_type = torch.load(osp.join(path, 'edge_type.pt'))
    edge_index = torch.load(osp.join(path, 'edge_index.pt'))
    num_for_h=torch.load('data/Twibot-20/num_properties_tensor.pt')
    num_for_h=torch.cat((torch.zeros((229580-11826,num_for_h.shape[-1])),num_for_h),dim=0)
    if dataset_name == 'Twibot-20':
        labels = torch.cat([labels, torch.empty(217754, dtype=torch.long).fill_(2)])
        train_idx = torch.arange(0, 8278)
        val_idx = torch.arange(8278, 8278 + 2365)
        test_idx = torch.arange(8278+2365, 8278 + 2365 + 1183)
    else:
        train_idx = torch.load(osp.join(path, 'train_idx.pt'))
        val_idx = torch.load(osp.join(path, 'val_idx.pt'))
        test_idx = torch.load(osp.join(path, 'test_idx.pt'))
    return Data(edge_index=edge_index,
                edge_type=edge_type,
                y=labels,
                num_for_h=num_for_h,
                des_embedding=des_embedding,
                tweet_embedding=tweet_embedding,
                num_property_embedding=num_property_embedding,
                cat_property_embedding=cat_property_embedding,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                num_nodes=des_embedding.shape[0])







def null_metrics():
    return {
        'acc': 0.0,
        'f1-score': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'mcc': 0.0,
        'roc-auc': 0.0,
        'pr-auc': 0.0
    }


def calc_metrics(y, pred):
    assert y.dim() == 1 and pred.dim() == 2
    if torch.any(torch.isnan(pred)):
        metrics = null_metrics()
        plog = ''
        for key, value in metrics.items():
            plog += ' {}: {:.6}'.format(key, value)
        return metrics, plog
    pred = func.softmax(pred, dim=-1)
    pred_label = torch.argmax(pred, dim=-1)
    pred_score = pred[:, -1]
    y = y.to('cpu').numpy().tolist()
    pred_label = pred_label.to('cpu').tolist()
    pred_score = pred_score.to('cpu').tolist()
    precision, recall, _thresholds = precision_recall_curve(y, pred_score)
    metrics = {
        'acc': accuracy_score(y, pred_label),
        'f1-score': f1_score(y, pred_label),
        'precision': precision_score(y, pred_label),
        'recall': recall_score(y, pred_label),
        'mcc': matthews_corrcoef(y, pred_label),
        'roc-auc': roc_auc_score(y, pred_score),
        'pr-auc': auc(recall, precision)
    }
    plog = ''
    for key in ['acc', 'f1-score', 'mcc']:
        plog += ' {}: {:.6}'.format(key, metrics[key])
    return metrics, plog