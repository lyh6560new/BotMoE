import os
import numpy as np
import pandas as pd
import torch
# os.environ['CUDA_VISIBLE_DEVICE'] = '3'
import math
from tqdm import tqdm
from torch import nn
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

    
class MLPclassifier(nn.Module):
    def __init__(self,
                 input_dim=728,
                 output_size=2,
                 hidden_dim=128,
                 dropout=0.3,all_mode=False):
        super(MLPclassifier, self).__init__()
        self.dropout = dropout
        
        
        self.all_mode=all_mode
        if(all_mode):
            self.linear_relu_tweet = nn.Sequential(
            nn.Linear(input_dim*2, hidden_dim),
            nn.LeakyReLU()
        )
        else:
            self.pre_model1 = nn.Linear(input_dim, input_dim // 2)
            self.pre_model2 = nn.Linear(input_dim, input_dim // 2)
            self.linear_relu_tweet = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU()
        )

        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(hidden_dim, output_size)
    
    def forward(self,tweet_feature, des_feature):
        if(not self.all_mode):
            pre1 = self.pre_model1(tweet_feature)
            pre2 = self.pre_model2(des_feature)
        else:
            pre1 = tweet_feature
            pre2 = des_feature
        x = torch.cat((pre1,pre2), dim=1)
        x = self.linear_relu_tweet(x)
        # x = self.linear_relu(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

'''
class MLPclassifier_aug(nn.Module):
    def __init__(self):
        super(MLPclassifier_aug, self).__init__()
        self.dropout = dropout
        #self.lm_model=LModel()

        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(hidden_dim, output_size)
    
    def forward(self,text):
        x,att=self.lm_model(text)
        x = self.dropout(x)
        x=self.classifier(x[:,0])
        return x
'''
class RobertaTrianer:
    def __init__(self,
                 train_loader,
                 val_loader,
                 test_loader,
                 epochs=100,
                 input_dim=768,
                 hidden_dim=128,
                 dropout=0.3,
                 activation='relu',
                 optimizer=torch.optim.Adam,
                 weight_decay=1e-6,
                 lr=1e-5,
                 device='cuda:0'):
        self.epochs = epochs
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        
        self.model = MLPclassifier(input_dim=self.input_dim, hidden_dim=self.hidden_dim, dropout=dropout)
        self.device = device
        self.model.to(self.device)
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_func = nn.CrossEntropyLoss()
        
    def train(self):
        train_loader = self.train_loader
        for epoch in range(self.epochs):
            self.model.train()
            loss_avg = 0
            preds = []
            preds_auc = []
            labels = []
            with tqdm(train_loader) as progress_bar:
                for batch in progress_bar:
                    pred = self.model(batch[0], batch[1])
                    loss = self.loss_func(pred, batch[2])
                    loss_avg += loss
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    progress_bar.set_description(desc=f'epoch={epoch}')
                    progress_bar.set_postfix(loss=loss.item())
                    
                    preds.append(pred.argmax(dim=-1).cpu().numpy())
                    preds_auc.append(pred[:, 1].detach().cpu().numpy())
                    labels.append(batch[2].cpu().numpy())
            
            preds = np.concatenate(preds, axis=0)
            preds_auc = np.concatenate(preds_auc, axis=0)
            labels = np.concatenate(labels, axis=0)
            loss_avg = loss_avg / len(train_loader)   
            print('{' + f'loss={loss_avg.item()}' + '}' + 'eval=', end='')
            eval(preds_auc, preds, labels)     
            self.valid()
            acc=self.test()
        torch.save(self.model,save_pth+f'acc_{acc}.pth')
        
    @torch.no_grad()
    def valid(self):
        self.model.eval()
        preds = []
        preds_auc = []
        labels = []
        val_loader = self.val_loader
        for batch in val_loader:
            pred = self.model(batch[0], batch[1])
            preds.append(pred.argmax(dim=-1).cpu().numpy())
            preds_auc.append(pred[:,1].detach().cpu().numpy())
            labels.append(batch[2].cpu().numpy())
        
        preds = np.concatenate(preds, axis=0)
        preds_auc = np.concatenate(preds_auc, axis=0)
        labels = np.concatenate(labels, axis=0)
        
        eval(preds_auc, preds, labels)
        
    @torch.no_grad()
    def test(self):
        self.model.eval()
        preds = []
        preds_auc = []
        labels = []
        test_loader = self.test_loader
        for batch in test_loader:
            pred = self.model(batch[0], batch[1])
            preds.append(pred.argmax(dim=-1).cpu().numpy())
            preds_auc.append(pred[:,1].detach().cpu().numpy())
            labels.append(batch[2].cpu().numpy())
            
        preds = np.concatenate(preds, axis=0)
        labels = np.concatenate(labels, axis=0)
        preds_auc = np.concatenate(preds_auc, axis=0)
        
        
        return eval(preds_auc, preds, labels)
  
     
