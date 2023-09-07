import torch
from torch import nn
import torch.nn.functional as F
from moe_all import MoE,MLP,MoE_receive_gate
class DeeProBot(nn.Module):
    def __init__(self,des_size=50,num_prop_size=9,dropout=0.1):
        super(DeeProBot, self).__init__()
        self.dropout = dropout
        self.des_size = des_size
        self.num_prop_size = num_prop_size
        
        self.lstm =nn.Sequential(
            nn.LSTM(input_size=self.des_size,hidden_size=32,num_layers=2,batch_first=True),
        )
        
        self.linear1 =nn.Sequential(
            nn.Linear(32+self.num_prop_size,128),
            nn.ReLU()
        )
        
        self.linear2 =nn.Sequential(
            nn.Linear(128,32),
            nn.ReLU()
        )
        
        self.output =nn.Linear(32,2)
        
    def forward(self,des,num_prop):
        des_out=self.lstm(des)[0]
        des_out=F.relu(des_out.sum(dim=1))
        x=torch.cat((des_out,num_prop),dim=1)
        
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.linear1(x)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.linear2(x)
        x=self.output(x)
        
        return x

class DeeProBot_MoE(nn.Module):
    def __init__(self,num_prop_size=9,dropout=0.1,expert_size=2,k=1,output_size=2,moe_out=32):
        super(DeeProBot_MoE, self).__init__()
        self.dropout = dropout
        #self.des_size = des_size
        self.num_prop_size = num_prop_size

        self.moe=MoE(self.num_prop_size,moe_out,expert_size,128,model=MLP,k=k)
        self.output =nn.Linear(moe_out,output_size)
        
    def forward(self,num_prop,cat_prop):
        #des_out=self.lstm(des)[0]
        #des_out=des
        #des_out=F.relu(des_out.sum(dim=1))
        #x=torch.cat((num_prop,cat_prop),dim=1)
        x=num_prop
        
        x=F.dropout(x,p=self.dropout,training=self.training)
        x,loss=self.moe(x)
        x=self.output(x)
        
        return x,loss

class DeeProBot_MoE_gate(nn.Module):
    def __init__(self,x_gate_size,num_prop_size=9,dropout=0.1,expert_size=2,k=1,output_size=2,moe_out=32):
        super(DeeProBot_MoE_gate, self).__init__()
        self.dropout = dropout
        #self.des_size = des_size
        self.num_prop_size = num_prop_size

        self.moe=MoE_receive_gate(self.num_prop_size,moe_out,expert_size,128,model=MLP,k=k,special_for_gate=4*num_prop_size)
        self.output =nn.Linear(moe_out,output_size)
        
    def forward(self,num_prop,cat_prop,gate):
        #des_out=self.lstm(des)[0]
        #des_out=des
        #des_out=F.relu(des_out.sum(dim=1))
        #x=torch.cat((num_prop,cat_prop),dim=1)
        x=num_prop
        gate=gate[:11826]
        x=F.dropout(x,p=self.dropout,training=self.training)
        x,loss=self.moe(x,gate)
        x=self.output(x)
        
        return x,loss


class DeeProBot_MoE_bl(nn.Module):
    def __init__(self,num_prop_size=9,dropout=0.1,expert_size=2,k=1,output_size=2,moe_out=32,loss_cof=1e-2):
        super(self.__class__, self).__init__()
        self.dropout = dropout
        #self.des_size = des_size
        self.loss_cof=loss_cof
        self.num_prop_size = num_prop_size

        self.moe=MoE(self.num_prop_size,moe_out,expert_size,128,model=MLP,k=k)
        self.output =nn.Linear(moe_out,output_size)
        
    def forward(self,num_prop,cat_prop):
        #des_out=self.lstm(des)[0]
        #des_out=des
        #des_out=F.relu(des_out.sum(dim=1))
        #x=torch.cat((num_prop,cat_prop),dim=1)
        x=num_prop
        
        x=F.dropout(x,p=self.dropout,training=self.training)
        x,loss=self.moe(x,self.loss_cof)
        x=self.output(x)
        
        return x,loss