import random
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import logging
from torch.utils.data import Dataset
import os
def set_random_seed():
    seed=np.random.randint(0,1e9)
    print(seed) 
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    return seed

def set_seed(seed:int):
    #seed=np.random.randint(0,1e9)
    print(seed) 
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    return seed


def set_logger(file,exp_name,root='MoE/mixture-of-experts/twibot-20/'):
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler(root+file)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info(exp_name)
    return logger


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def init_weights(m):
    if type(m)==nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        #nn.init.kaiming_uniform_(m.weight,a=math.sqrt(5))

def init_weights_con(m):
    if type(m)==nn.Linear:
        nn.init.constant_(m.weight,0.2)
        #nn.init.kaiming_uniform_(m.weight)



class Twibot22(Dataset):
    def __init__(self,root='./Data/',device='cpu'):
        self.root = root
        self.device = device
        
        
    def load_labels(self):
        #print('Loading labels...',end='   ')
        path=self.root+'label.pt'
        labels=torch.load(self.root+"label.pt").to(self.device)
        #print('Finished')
        
        return labels
    
    def Des_Preprocess(self):
        #print('Loading raw feature1...',end='   ')
        path=self.root+'description.npy'
        description=np.load(path,allow_pickle=True)
        #print('Finished')
        return description

    def Des_embbeding(self):
        #print('Running feature1 embedding')
        path=self.root+"des_tensor.pt"
        des_tensor=torch.load(self.root+"des_tensor.pt").to(self.device)
        #print('Finished')
        return des_tensor
    
    def tweets_preprocess(self):
        #print('Loading raw feature2...',end='   ')
        path=self.root+'tweets.npy'
        tweets=np.load(path,allow_pickle=True)
        #print('Finished')
        return tweets
    
    def tweets_embedding(self):
        #print('Running feature2 embedding')
        path=self.root+"tweets_tensor.pt"
        tweets_tensor=torch.load(self.root+"tweets_tensor.pt").to(self.device)
        #print('Finished')
        return tweets_tensor
    
    def num_prop_preprocess(self):
        #print('Processing feature3...',end='   ')
        path0=self.root+'num_properties_tensor.pt'
        num_prop=torch.load(self.root+"num_properties_tensor.pt").to(self.device)
        #print('Finished')
        return num_prop
    
    def cat_prop_preprocess(self):
        #print('Processing feature4...',end='   ')
        path=self.root+'cat_properties_tensor.pt'
        category_properties=torch.load(self.root+"cat_properties_tensor.pt").to(self.device)
        #print('Finished')
        return category_properties
    
    def Build_Graph(self):
        #print('Building graph',end='   ')
        path=self.root+'edge_index.pt'
        edge_index=torch.load(self.root+"edge_index.pt").to(self.device)
        edge_type=torch.load(self.root+"edge_type.pt").to(self.device)
        #print('Finished')
        return edge_index,edge_type
    
    def train_val_test_mask(self):
        if self.root=='/data1/whr/lyh/data/':
            train_idx=range(8278)
            val_idx=range(8278,8278+2365)
            test_idx=range(8278+2365,8278+2365+1183)
        else:
            train_idx=torch.load(self.root+'train_idx.pt')
            val_idx=torch.load(self.root+'val_idx.pt')
            test_idx=torch.load(self.root+'test_idx.pt')
            
        return train_idx,val_idx,test_idx
        
        
    def dataloader(self):
        labels=self.load_labels()
        #self.Des_Preprocess()
        des_tensor=self.Des_embbeding()
        #self.tweets_preprocess()
        tweets_tensor=self.tweets_embedding()
        num_prop=self.num_prop_preprocess()
        category_prop=self.cat_prop_preprocess()
        edge_index,edge_type=self.Build_Graph()
        num_for_h=torch.load('data/Twibot-20/num_properties_tensor.pt').to(self.device)
        train_idx,val_idx,test_idx=self.train_val_test_mask()
        return des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,labels,train_idx,val_idx,test_idx,num_for_h

class MaskDataset(Dataset):
    def __init__(self, idx):
        self.idx = torch.tensor(idx, dtype=torch.long)

    def __len__(self):
        return self.idx.shape[0]

    def __getitem__(self, index):
        return self.idx[index]
    
    
class Twi20manipulated(Dataset):
    def __init__(self,p,exp,root='/data3/whr/lyh/MoE/mixture-of-experts/BotRGCN/twibot_20/processed_data/manipulated_data',device='cpu'):
        self.root = root+f'/p_{p}/'
        self.device = device
        self.exp=exp
    def train_val_test_mask(self):
        train_idx=range(8278)
        val_idx=range(8278,8278+2365)
        test_idx=range(8278+2365,8278+2365+1183)
            
        return train_idx,val_idx,test_idx
    def load_labels(self):
        #print('Loading labels...',end='   ')
        path=self.root+'label.pt'
        labels=torch.load(self.root+"label.pt").to(self.device)
        #print('Finished')
        
        return labels
    
    def Des_Preprocess(self):
        #print('Loading raw feature1...',end='   ')
        path=self.root+'description.npy'
        description=np.load(path,allow_pickle=True)
        #print('Finished')
        return description

    def Des_embbeding(self):
        #print('Running feature1 embedding')
        path=self.root+"des_tensor.pt"
        des_tensor=torch.load(self.root+"des_tensor.pt").to(self.device)
        #print('Finished')
        return des_tensor
    
    def tweets_preprocess(self):
        #print('Loading raw feature2...',end='   ')
        path=self.root+'tweets.npy'
        tweets=np.load(path,allow_pickle=True)
        #print('Finished')
        return tweets
    
    def tweets_embedding(self):
        #print('Running feature2 embedding')
        path=self.root+"tweets_tensor.pt"
        tweets_tensor=torch.load(self.root+"tweets_tensor.pt").to(self.device)
        #print('Finished')
        return tweets_tensor
    
    def num_prop_preprocess(self):
        #print('Processing feature3...',end='   ')
        path0=self.root+'num_properties_tensor.pt'
        num_prop=torch.load(self.root+"num_properties_tensor.pt").to(self.device)
        #print('Finished')
        return num_prop
    
    def cat_prop_preprocess(self):
        #print('Processing feature4...',end='   ')
        path=self.root+'cat_properties_tensor.pt'
        category_properties=torch.load(self.root+"cat_properties_tensor.pt").to(self.device)
        #print('Finished')
        return category_properties
    
    def Build_Graph(self):
        #print('Building graph',end='   ')
        path=self.root+'edge_index.pt'
        
        edge_index=torch.load(path).to(self.device)
        path=self.root+'edge_type.pt'
        edge_type=torch.load(path).to(self.device)
        #print('Finished')
        return edge_index,edge_type.long()
    def dataloader(self):
        labels=self.load_labels()
        #self.Des_Preprocess()
        if(self.exp=='text'):
            des_tensor=self.Des_embbeding()
            #self.tweets_preprocess()
            tweets_tensor=self.tweets_embedding()
        else:
            des_tensor=torch.load('/data3/whr/lyh/MoE/mixture-of-experts/BotRGCN/twibot_20/processed_data/'+"des_tensor.pt").to(self.device)
            #self.tweets_preprocess()
            tweets_tensor=torch.load('/data3/whr/lyh/MoE/mixture-of-experts/BotRGCN/twibot_20/processed_data/'+'tweets_tensor.pt').to(self.device)
            
        if(self.exp=='meta'):
            num_for_h=torch.load(self.root+'num_for_h.pt').to(self.device)
            num_prop=self.num_prop_preprocess()
            category_prop=self.cat_prop_preprocess()
        else:
            num_for_h=torch.load('/data3/whr/lyh/data/Twibot-20/num_properties_tensor.pt').to(self.device)
            num_prop=torch.load('/data3/whr/lyh/MoE/mixture-of-experts/BotRGCN/twibot_20/processed_data/'+"num_properties_tensor.pt").to(self.device)
            category_prop=torch.load('/data3/whr/lyh/MoE/mixture-of-experts/BotRGCN/twibot_20/processed_data/'+"cat_properties_tensor.pt").to(self.device)
        if(self.exp=='graph'):
            
            edge_index,edge_type=self.Build_Graph()
        else:
            edge_index = torch.load('/data3/whr/lyh/MoE/mixture-of-experts/BotRGCN/twibot_20/processed_data/'+"edge_index.pt").to(self.device)
            edge_type = torch.load('/data3/whr/lyh/MoE/mixture-of-experts/BotRGCN/twibot_20/processed_data/'+"edge_type.pt").to(self.device)
        train_idx,val_idx,test_idx=self.train_val_test_mask()
        return des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,labels,train_idx,val_idx,test_idx,num_for_h

    