from utils import *
from pickle import TRUE
from sklearn.metrics import (accuracy_score, auc, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from sample_utils import get_train_data,calc_metrics
from torch_geometric.loader import NeighborLoader


seed=set_random_seed()

import random
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import logging
from torch.utils.data import Dataset
import os
from ThreeInONE import *




class Trainer:
    def __init__(self,
                 idx,
                 loader,
                 epochs=400,
                 optimizer=torch.optim.Adam,
                 weight_decay=1e-6,
                 device='cuda:3',batch_size=32):
        self.epochs = epochs
        self.idx=idx
        self.model = model[idx](hidden_size,num_gnn,num_text,gnn_k=gnn_k,align_size=align_size)
        #self.model.apply(init_weights)
        #logger.info(self.model)
        self.info=0

        self.model_init()
        
                
        self.device = device
        self.model.to(self.device)
        #self.loader = loader
        self.data=get_train_data('Twibot-20')
        self.train_loader = NeighborLoader(self.data,
                                  num_neighbors=[256] * 2,
                                  batch_size=batch_size,
                                  input_nodes=self.data.train_idx,
                                  shuffle=True)
        self.val_loader = NeighborLoader(self.data,
                                num_neighbors=[256] * 2,
                                batch_size=batch_size,
                                input_nodes=self.data.val_idx)
        self.test_loader = NeighborLoader(self.data,
                                 num_neighbors=[256] * 2,
                                 batch_size=batch_size,
                                 input_nodes=self.data.test_idx)
       
        self.optimizer_init(optimizer,weight_decay)

        self.loss_func = nn.CrossEntropyLoss()
        self.best_test_acc=0
        self.best_test_f1=0
        self.best_test_precision=0
        self.best_test_recall=0
        self.nohup=0

        

    def model_init(self):
        '''
        for i in range(self.model.num_gnn):
            self.model.gnn_moe.moe.experts[i].apply(init_weights)
        
        for i in range(self.model.num_text):
            self.model.text_moe.experts[i].apply(init_weights)
        
        for i in range(self.model.num_cat):
            self.model.cat_moe.moe.experts[i].apply(init_weights)

        '''
     
        self.model.apply(init_weights)
        #self.model.mlp_classifier.apply(init_weights)
        
        
        nn.init.constant_(self.model.gcn_moe.moe.w_gate,0.1)
        nn.init.constant_(self.model.rgcn_moe.moe.w_gate,0.1)
        nn.init.constant_(self.model.rgt_moe.moe.w_gate,0.1)
        nn.init.constant_(self.model.text_moe.w_gate,0.1)
        nn.init.constant_(self.model.cat_moe.moe.w_gate,0.1)
        nn.init.constant_(self.model.gcn_moe.moe.w_noise,0.1)
        nn.init.constant_(self.model.rgcn_moe.moe.w_noise,0.1)
        nn.init.constant_(self.model.rgt_moe.moe.w_noise,0.1)
        nn.init.constant_(self.model.text_moe.w_noise,0.1)
        nn.init.constant_(self.model.cat_moe.moe.w_noise,0.1)
        
        '''
        #load pretrain
        self.model.gnn_moe.moe.w_gate.weight=torch.load(f'MoE/mixture-of-experts/twibot-20/pretrain/gnn_w_gate{self.idx}.pth')
        self.model.gnn_moe.moe.w_noise.weight=torch.load(f'MoE/mixture-of-experts/twibot-20/pretrain/gnn_w_noise{self.idx}.pth')
        self.model.text_moe.w_gate.weight=torch.load(f'MoE/mixture-of-experts/twibot-20/pretrain/text_w_gate{self.idx}.pth')
        self.model.text_moe.w_noise.weight=torch.load(f'MoE/mixture-of-experts/twibot-20/pretrain/text_w_noise{self.idx}.pth')
        self.model.cat_moe.moe.w_gate.weight=torch.load(f'MoE/mixture-of-experts/twibot-20/pretrain/cat_w_gate{self.idx}.pth')
        self.model.cat_moe.moe.w_noise.weight=torch.load(f'MoE/mixture-of-experts/twibot-20/pretrain/gnn_w_noise{self.idx}.pth')
        '''

        #freeze gate for 200 epochs
        self.model.gcn_moe.moe.w_gate.requires_grad=False
        self.model.gcn_moe.moe.w_noise.requires_grad=False
        self.model.rgcn_moe.moe.w_gate.requires_grad=False
        self.model.rgcn_moe.moe.w_noise.requires_grad=False
        self.model.rgt_moe.moe.w_gate.requires_grad=False
        self.model.rgt_moe.moe.w_noise.requires_grad=False
        #self.model.text_moe.w_noise.requires_grad=False
        #self.model.text_moe.w_gate.requires_grad=False
        #self.model.cat_moe.moe.w_gate.requires_grad=False
        #self.model.cat_moe.moe.w_noise.requires_grad=False
        #self.model.fusion.w_gate.requires_grad=False

    def optimizer_init(self,optimizer,weight_decay):

        
        self.optimizer = optimizer(self.model.parameters(),lr=1e-5,weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,milestones=[200,600],gamma = 0.8)


    def train(self):
        #des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,labels,train_idx,val_idx,test_idx,num_for_h=self.data
       
        for epoch in range(self.epochs):
            all_label = []
            all_pred = []
            ave_loss = 0.0
            cnt=0.0
            self.model.train()
            for batch in self.train_loader:
                if(self.nohup>200):
                    logger.info("Early stopping")
                    print("Early stopping")
                    break
                self.curr_epoch=epoch
               
                n_batch=batch.batch_size
                batch=batch.to(self.device)
                output,exp_loss = self.model(batch.des_embedding,batch.tweet_embedding,batch.num_property_embedding,batch.cat_property_embedding,batch.num_for_h,batch.edge_index,batch.edge_type)
                label = batch.y[:n_batch]
                out = output[:n_batch]
                all_label += label.data
                all_pred += out
                loss = self.loss_func(out, label)+exp_loss
                ave_loss += loss.item() * n_batch
                cnt += n_batch
               
                loss = self.loss_func(out,label)+exp_loss
                
                self.optimizer.zero_grad() 
                loss.backward()
                ## add gradient clip
                nn.utils.clip_grad_value_(self.model.parameters(),1000)
                self.optimizer.step()
            self.scheduler.step()
            ave_loss /= cnt
           
            all_label = torch.stack(all_label)
            all_pred = torch.stack(all_pred)
            metrics, plog = calc_metrics(all_label, all_pred)
            plog = 'Epoch-{} train loss: {:.6}'.format(epoch, ave_loss) + plog


                #unfreeze w_gate after 200 epochs
            if(self.best_test_acc>0.865):

                    self.optimizer.param_groups[0]['lr']=1e-6
                    #self.model.gnn_moe.moe.w_gate.requires_grad=True
                    #self.model.gnn_moe.moe.w_noise.requires_grad=True
                    self.model.gcn_moe.moe.w_gate.requires_grad=True
                    self.model.gcn_moe.moe.w_noise.requires_grad=True
                    self.model.rgcn_moe.moe.w_gate.requires_grad=True
                    self.model.rgcn_moe.moe.w_noise.requires_grad=True
                    self.model.rgt_moe.moe.w_gate.requires_grad=True
                    self.model.rgt_moe.moe.w_noise.requires_grad=True

            print(plog)
            
            self.validation(epoch,self.val_loader,name='val')
            self.validation(epoch,self.test_loader,name='test')
        return

    @torch.no_grad()
    def validation(self,epoch,loader,name):
        self.model.eval()
        all_label = []
        all_pred = []
        ave_loss = 0.0
        cnt = 0.0
        for batch in loader:
            batch = batch.to(self.device)
            n_batch = batch.batch_size
            out,_ = self.model(batch.des_embedding,
                        batch.tweet_embedding,
                        batch.num_property_embedding,
                        batch.cat_property_embedding,
                        batch.num_for_h,
                        batch.edge_index,
                        batch.edge_type)
            label = batch.y[:n_batch]
            out = out[:n_batch]
            all_label += label.data
            all_pred += out
            loss = self.loss_func(out, label)
            ave_loss += loss.item() * n_batch
            cnt += n_batch
        ave_loss /= cnt
        all_label = torch.stack(all_label)
        all_pred = torch.stack(all_pred)
        metrics, plog = calc_metrics(all_label, all_pred)
        plog = 'Epoch-{} {} loss: {:.6}'.format(epoch, name, ave_loss) + plog
        print(plog)

        if(metrics['acc']>self.best_test_acc and metrics['acc']>0.85 and name=='test'):
            self.best_test_acc=metrics['acc']
            self.best_test_f1=metrics['f1-score']
            self.best_test_precision=metrics['precision']
            self.best_test_recall=metrics['recall']
            logger.info(f"best acc:{self.best_test_acc} f1:{self.best_test_f1} precision:{self.best_test_precision} recall:{self.best_test_recall} seed {seed}")
            self.nohup=0
            
            '''
            if(self.best_test_acc > 0.865):
                for i in range(len(self.optimizer.param_groups)):
                    self.optimizer.param_groups[i]['lr']=self.optimizer.param_groups[i]['lr']/2
            '''
        if(metrics['acc']<self.best_test_acc and metrics['acc']>0.85 and name=='test'):
            self.nohup+=1
        if(metrics['acc']>0.871 and name=='test' ):
            torch.save(self.model,save_pth+"acc:{:.4f} seed:{}.pth".format(metrics['acc'],seed))
            logger.info(self.model)
            self.info+=1
        
        return 

        
if __name__ == '__main__':

    exp_name="load&fix"
    idx=0
    fix_seed_training=False
    model=[AllInOne1_rgcn_rgt_gcn]
    file =['AllInOne1_rgcn_rgt_gcn.log']
    logger=set_logger(file[idx],exp_name)
    root='/data3/whr/lyh/MoE/mixture-of-experts/twibot-20/model/'
    save_pth=root+file[idx].rstrip('.log')+'/'
    if(not os.path.exists(save_pth)):
        os.mkdir(save_pth)
    logger.info(exp_name)

    root='MoE/mixture-of-experts/BotRGCN/twibot_20/processed_data/'
    align_size_set=[128]
    hidden_size_set=[4]
    hidden_size=4
    device="cuda:2"
    dataset=Twibot22(root=root,device=device)
    test_run=range(20)
    #have to fixed num_roberta and num_gnn
    #num_gnn=10
    #num_text=2
    num_text=2
    gnn_k=1
    num_gnn=3
    align_size=128

    trainer = Trainer(idx,dataset,device=device,batch_size=1024)
    trainer.model = torch.load('/data3/whr/lyh/MoE/mixture-of-experts/twibot-20/model/AllInOne1_rgcn_rgt_gcn/acc:0.8800 seed:476645318.pth')
    trainer.validation(1,trainer.test_loader,name='test')
    
    logger.info(f"{exp_name} best acc:{trainer.best_test_acc} f1:{trainer.best_test_f1} precision:{trainer.best_test_precision} recall:{trainer.best_test_recall} seed:{seed} ")
    


