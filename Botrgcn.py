import torch
from torch import nn
from torch_geometric.nn.conv import RGCNConv,GCNConv,GATConv
from torch_geometric.nn import TransformerConv
import torch.nn.functional as F
from moe_all import MoE,MoE_share_gate
from layer_HGN  import SimpleHGN

class BotRGCN(nn.Module):
    def __init__(self,input_size=0,output_size=0,hidden_size=0,des_size=768,tweet_size=768,num_prop_size=5,cat_prop_size=3,embedding_dimension=128,dropout=0.4):
        super(BotRGCN, self).__init__()
        self.dropout = dropout
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(tweet_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(num_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(cat_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        
        self.linear_relu_input=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        
        self.rgcn=RGCNConv(embedding_dimension,embedding_dimension,num_relations=2)
        
        self.linear_relu_output1=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2=nn.Linear(embedding_dimension,2)
        
        
        
    def forward(self,des,tweet,num_prop,cat_prop,edge_index,edge_type):
        d=self.linear_relu_des(des)
        t=self.linear_relu_tweet(tweet)
        n=self.linear_relu_num_prop(num_prop)
        c=self.linear_relu_cat_prop(cat_prop)
        x=torch.cat((d,t,n,c),dim=1)
        
        x=self.linear_relu_input(x)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        x=self.linear_relu_output1(x)
        x=self.linear_output2(x)
            
        return x
            
            
class BotRGCN1(nn.Module):
    def __init__(self,des_size=768,tweet_size=768,num_prop_size=6,cat_prop_size=11,embedding_dimension=128,dropout=0.3):
        super(BotRGCN1, self).__init__()
        self.dropout = dropout
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension)),
            nn.LeakyReLU()
        )
        ''''
        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(tweet_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(num_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(cat_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        '''
        self.linear_relu_input=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2=nn.Linear(embedding_dimension,2)
        
        self.rgcn=RGCNConv(embedding_dimension,embedding_dimension,num_relations=2)
        
    def forward(self,des,tweet,num_prop,cat_prop,edge_index,edge_type):
        d=self.linear_relu_des(des)
        '''
        t=self.linear_relu_tweet(tweet)
        n=self.linear_relu_num_prop(num_prop)
        c=self.linear_relu_cat_prop(cat_prop)
        '''
        x=d
        
        x=self.linear_relu_input(x)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        x=self.linear_relu_output1(x)
        x=self.linear_output2(x)
            
        return x
            
            
class BotRGCN2(nn.Module):
    def __init__(self,des_size=768,tweet_size=768,num_prop_size=6,cat_prop_size=11,embedding_dimension=128,dropout=0.3):
        super(BotRGCN2, self).__init__()
        self.dropout = dropout
        '''
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension)),
            nn.LeakyReLU()
        )
        '''
        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(tweet_size,int(embedding_dimension)),
            nn.LeakyReLU()
        )
        '''
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(num_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(cat_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        '''
        self.linear_relu_input=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2=nn.Linear(embedding_dimension,2)
        
        self.rgcn=RGCNConv(embedding_dimension,embedding_dimension,num_relations=2)
        
    def forward(self,des,tweet,num_prop,cat_prop,edge_index,edge_type):
        '''
        d=self.linear_relu_des(des)
        '''
        t=self.linear_relu_tweet(tweet)
        '''
        n=self.linear_relu_num_prop(num_prop)
        c=self.linear_relu_cat_prop(cat_prop)
        '''
        x=t
        
        x=self.linear_relu_input(x)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        x=self.linear_relu_output1(x)
        x=self.linear_output2(x)
            
        return x
            
            
class BotRGCN3(nn.Module):
    def __init__(self,des_size=768,tweet_size=768,num_prop_size=6,cat_prop_size=11,embedding_dimension=128,dropout=0.3):
        super(BotRGCN3, self).__init__()
        self.dropout = dropout
        '''
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension)),
            nn.LeakyReLU()
        )

        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(tweet_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        '''
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(num_prop_size,int(embedding_dimension)),
            nn.LeakyReLU()
        )
        '''
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(cat_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        '''
        self.linear_relu_input=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2=nn.Linear(embedding_dimension,2)
        
        self.rgcn=RGCNConv(embedding_dimension,embedding_dimension,num_relations=2)
        
    def forward(self,des,tweet,num_prop,cat_prop,edge_index,edge_type):
        '''
        d=self.linear_relu_des(des)
        t=self.linear_relu_tweet(tweet)
        '''
        n=self.linear_relu_num_prop(num_prop)
        '''
        c=self.linear_relu_cat_prop(cat_prop)
        '''
        x=n
        
        x=self.linear_relu_input(x)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        x=self.linear_relu_output1(x)
        x=self.linear_output2(x)
            
        return x
            
            
class BotRGCN4(nn.Module):
    def __init__(self,des_size=768,tweet_size=768,num_prop_size=6,cat_prop_size=11,embedding_dimension=128,dropout=0.3):
        super(BotRGCN4, self).__init__()
        self.dropout = dropout
        '''
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension)),
            nn.LeakyReLU()
        )

        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(tweet_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(num_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        '''
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(cat_prop_size,int(embedding_dimension)),
            nn.LeakyReLU()
        )
        self.linear_relu_input=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2=nn.Linear(embedding_dimension,2)
        
        self.rgcn=RGCNConv(embedding_dimension,embedding_dimension,num_relations=2)
        
    def forward(self,des,tweet,num_prop,cat_prop,edge_index,edge_type):
        '''
        d=self.linear_relu_des(des)
        t=self.linear_relu_tweet(tweet)
        n=self.linear_relu_num_prop(num_prop)
        '''
        c=self.linear_relu_cat_prop(cat_prop)
        x=c
        
        x=self.linear_relu_input(x)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        x=self.linear_relu_output1(x)
        x=self.linear_output2(x)
            
        return x
    
class BotRGCN12(nn.Module):
    def __init__(self,des_size=768,tweet_size=768,num_prop_size=6,cat_prop_size=11,embedding_dimension=128,dropout=0.3):
        super(BotRGCN12, self).__init__()
        self.dropout = dropout
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension/2)),
            nn.LeakyReLU()
        )

        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(tweet_size,int(embedding_dimension/2)),
            nn.LeakyReLU()
        )
        '''
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(num_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(cat_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        '''
        self.linear_relu_input=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2=nn.Linear(embedding_dimension,2)
        
        self.rgcn=RGCNConv(embedding_dimension,embedding_dimension,num_relations=2)
        
    def forward(self,des,tweet,num_prop,cat_prop,edge_index,edge_type):
        d=self.linear_relu_des(des)
        t=self.linear_relu_tweet(tweet)
        '''
        n=self.linear_relu_num_prop(num_prop)
        c=self.linear_relu_cat_prop(cat_prop)
        '''
        x=torch.cat((d,t),dim=1)
        
        x=self.linear_relu_input(x)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        x=self.linear_relu_output1(x)
        x=self.linear_output2(x)
            
        return x
    
class BotRGCN34(nn.Module):
    def __init__(self,des_size=768,tweet_size=768,num_prop_size=5,cat_prop_size=3,embedding_dimension=128,dropout=0.3):
        super(BotRGCN34, self).__init__()
        self.dropout = dropout
        '''
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension/2)),
            nn.LeakyReLU()
        )

        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(tweet_size,int(embedding_dimension/2)),
            nn.LeakyReLU()
        )
        '''
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(num_prop_size,int(embedding_dimension/2)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(cat_prop_size,int(embedding_dimension/2)),
            nn.LeakyReLU()
        )
        self.linear_relu_input=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2=nn.Linear(embedding_dimension,2)
        
        self.rgcn=RGCNConv(embedding_dimension,embedding_dimension,num_relations=2)
        
    def forward(self,des,tweet,num_prop,cat_prop,edge_index,edge_type):
        '''
        d=self.linear_relu_des(des)
        t=self.linear_relu_tweet(tweet)
        '''
        n=self.linear_relu_num_prop(num_prop)
        c=self.linear_relu_cat_prop(cat_prop)
        x=torch.cat((n,c),dim=1)
        
        x=self.linear_relu_input(x)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        x=self.linear_relu_output1(x)
        x=self.linear_output2(x)
            
        return x
    
class BotGCN(nn.Module):
    def __init__(self,des_size=768,tweet_size=768,num_prop_size=6,cat_prop_size=11,embedding_dimension=128,dropout=0.3):
        super(BotGCN, self).__init__()
        self.dropout = dropout
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(tweet_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(num_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(cat_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        
        self.linear_relu_input=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2=nn.Linear(embedding_dimension,2)
        
        self.gcn1=GCNConv(embedding_dimension,embedding_dimension)
        self.gcn2=GCNConv(embedding_dimension,embedding_dimension)
        
    def forward(self,des,tweet,num_prop,cat_prop,edge_index,edge_type):
        d=self.linear_relu_des(des)
        t=self.linear_relu_tweet(tweet)
        n=self.linear_relu_num_prop(num_prop)
        c=self.linear_relu_cat_prop(cat_prop)
        x=torch.cat((d,t,n,c),dim=1)
        
        x=self.linear_relu_input(x)
        x=self.gcn1(x,edge_index)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.gcn2(x,edge_index)
        x=self.linear_relu_output1(x)
        x=self.linear_output2(x)
            
        return x
    
class BotGAT(nn.Module):
    def __init__(self,des_size=768,tweet_size=768,num_prop_size=6,cat_prop_size=11,embedding_dimension=128,dropout=0.3):
        super(BotGAT, self).__init__()
        self.dropout = dropout
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(tweet_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(num_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(cat_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        
        self.linear_relu_input=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2=nn.Linear(embedding_dimension,2)
        
        self.gat1=GATConv(embedding_dimension,int(embedding_dimension/4),heads=4)
        self.gat2=GATConv(embedding_dimension,embedding_dimension)
        
    def forward(self,des,tweet,num_prop,cat_prop,edge_index,edge_type):
        d=self.linear_relu_des(des)
        t=self.linear_relu_tweet(tweet)
        n=self.linear_relu_num_prop(num_prop)
        c=self.linear_relu_cat_prop(cat_prop)
        x=torch.cat((d,t,n,c),dim=1)
        
        x=self.linear_relu_input(x)
        x=self.gat1(x,edge_index)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.gat2(x,edge_index)
        x=self.linear_relu_output1(x)
        x=self.linear_output2(x)
            
        return x
            

#implement BotRGCN with mlp
class BotRGCN_fmoe(nn.Module):
    def __init__(self,input_size=0,output_size=0,hidden_size=0,des_size=768,tweet_size=768,num_prop_size=5,cat_prop_size=3,embedding_dimension=128,dropout=0.4,expert_size=2,k=1):
        super(BotRGCN_fmoe, self).__init__()
        self.dropout = dropout
        self.embedding_dimension=embedding_dimension
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension/4)),
            nn.SELU()
        )

        self.linear_relu_tweet=nn.Sequential(
                nn.Linear(tweet_size,int(embedding_dimension/4)),
                nn.SELU()
            )
        self.linear_relu_num_prop=nn.Sequential(
                nn.Linear(num_prop_size,int(embedding_dimension/4)),
                nn.SELU()
            )
        self.linear_relu_cat_prop=nn.Sequential(
                nn.Linear(cat_prop_size,int(embedding_dimension/4)),
                nn.SELU()
            )
            
        self.linear_relu_input=nn.Sequential(
                nn.Linear(embedding_dimension,embedding_dimension),
                nn.SELU()
            )
            
        self.rgcn=RGCNConv(embedding_dimension,embedding_dimension,num_relations=2)
        

        self.moe=MoE(embedding_dimension,output_size,expert_size,embedding_dimension,model=MLP,k=k)
        
        
    def forward(self,des,tweet,num_prop,cat_prop,edge_index,edge_type):

        d=self.linear_relu_des(des)
        t=self.linear_relu_tweet(tweet)
        n=self.linear_relu_num_prop(num_prop)
        c=self.linear_relu_cat_prop(cat_prop)

            
        x=torch.cat((d,t,n,c),dim=1)
        
        x=self.linear_relu_input(x)
        
        self.rgcn.explain=False
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        #residual connection
        #x+=torch.cat((d,t,n,c),dim=1)
        x=self.moe(x)
            
        return x

class BotRGCN_fmoe_gate(nn.Module):
    def __init__(self,input_size=0,output_size=0,hidden_size=0,des_size=768,tweet_size=768,num_prop_size=5,cat_prop_size=3,embedding_dimension=128,dropout=0.4,expert_size=2,k=1):
        super(BotRGCN_fmoe_gate, self).__init__()
        self.dropout = dropout
        self.embedding_dimension=embedding_dimension
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension/4)),
            nn.SELU()
        )

        self.linear_relu_tweet=nn.Sequential(
                nn.Linear(tweet_size,int(embedding_dimension/4)),
                nn.SELU()
            )
        self.linear_relu_num_prop=nn.Sequential(
                nn.Linear(num_prop_size,int(embedding_dimension/4)),
                nn.SELU()
            )
        self.linear_relu_cat_prop=nn.Sequential(
                nn.Linear(cat_prop_size,int(embedding_dimension/4)),
                nn.SELU()
            )
            
        self.linear_relu_input=nn.Sequential(
                nn.Linear(embedding_dimension,embedding_dimension),
                nn.SELU()
            )
            
        self.rgcn=RGCNConv(embedding_dimension,embedding_dimension,num_relations=2)
        

        self.moe=MoE_share_gate(embedding_dimension,output_size,expert_size,embedding_dimension,model=MLP,k=k)
        
        
    def forward(self,des,tweet,num_prop,cat_prop,edge_index,edge_type):

        d=self.linear_relu_des(des)
        t=self.linear_relu_tweet(tweet)
        n=self.linear_relu_num_prop(num_prop)
        c=self.linear_relu_cat_prop(cat_prop)

            
        x=torch.cat((d,t,n,c),dim=1)
        
        x=self.linear_relu_input(x)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        #residual connection
        #x+=torch.cat((d,t,n,c),dim=1)
        x,loss,gate=self.moe(x)
            
        return x,loss,gate


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.LeakyReLU()
        #self.soft = nn.LeakyReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        #out = self.soft(out)
        return out
class MLP2(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP2, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.LeakyReLU()
        self.soft =  nn.Softmax(1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.soft(out)
        return out

class BotRGCN_fmoe1(nn.Module):
    def __init__(self,input_size=0,output_size=0,hidden_size=0,des_size=768,tweet_size=768,num_prop_size=5,cat_prop_size=3,embedding_dimension=128,dropout=0.4,expert_size=2,k=1):
        super(BotRGCN_fmoe1, self).__init__()
        self.dropout = dropout
        self.embedding_dimension=embedding_dimension
        self.linear_relu_input=nn.Sequential(
                nn.Linear(embedding_dimension,embedding_dimension),
                nn.SELU()
            )
            
        self.rgcn=RGCNConv(embedding_dimension,embedding_dimension,num_relations=2)
        self.moe=MoE(embedding_dimension,output_size,expert_size,embedding_dimension,model=MLP,k=k)
        
        
    def forward(self,des,tweet,num_prop,cat_prop,edge_index,edge_type):

        d,t,n,c=des,tweet,num_prop,cat_prop
        x=torch.cat((d,t,n,c),dim=1)
        
        x=self.linear_relu_input(x)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        #residual connection
        #x+=torch.cat((d,t,n,c),dim=1)
        x=self.moe(x)
            
        return x

class BotRGCN_fmoe2(nn.Module):
    def __init__(self,input_size=0,output_size=0,hidden_size=0,des_size=768,tweet_size=768,num_prop_size=5,cat_prop_size=3,embedding_dimension=128,dropout=0.4,expert_size=2,k=1):
        super(BotRGCN_fmoe2, self).__init__()
        self.dropout = dropout
        self.embedding_dimension=embedding_dimension
        self.linear_relu_input=nn.Sequential(
                nn.Linear(embedding_dimension,embedding_dimension),
                nn.SELU()
            )
            
        self.gcn1=GCNConv(embedding_dimension,embedding_dimension)
        self.gcn2=GCNConv(embedding_dimension,embedding_dimension)
        self.moe=MoE(embedding_dimension,output_size,expert_size,embedding_dimension,model=MLP,k=k)
        
        
    def forward(self,des,tweet,num_prop,cat_prop,edge_index,edge_type):

        d,t,n,c=des,tweet,num_prop,cat_prop
        x=torch.cat((d,t,n,c),dim=1)
        
        x=self.linear_relu_input(x)
        x=self.gcn1(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.gcn1(x,edge_index,edge_type)
        #residual connection
        #x+=torch.cat((d,t,n,c),dim=1)
        x=self.moe(x)
            
        return x

class BotRGCN_fmoe_double(nn.Module):
    def __init__(self,input_size=0,output_size=0,hidden_size=0,des_size=768,tweet_size=768,num_prop_size=5,cat_prop_size=3,embedding_dimension=128,dropout=0.4,expert_size=2,k=1):
        super(BotRGCN_fmoe_double, self).__init__()
        self.dropout = dropout
        self.embedding_dimension=embedding_dimension
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension/4)),
            nn.SELU()
        )

        self.linear_relu_tweet=nn.Sequential(
                nn.Linear(tweet_size,int(embedding_dimension/4)),
                nn.SELU()
            )
        self.linear_relu_num_prop=nn.Sequential(
                nn.Linear(num_prop_size,int(embedding_dimension/4)),
                nn.SELU()
            )
        self.linear_relu_cat_prop=nn.Sequential(
                nn.Linear(cat_prop_size,int(embedding_dimension/4)),
                nn.SELU()
            )
            
        self.linear_relu_input=nn.Sequential(
                nn.Linear(embedding_dimension,embedding_dimension),
                nn.SELU()
            )
            
        self.rgcn=RGCNConv(embedding_dimension,embedding_dimension,num_relations=2)
        self.rgcn2=RGCNConv(embedding_dimension,embedding_dimension,num_relations=2)
        self.linear_relu_input2=nn.Sequential(
                nn.Linear(embedding_dimension,embedding_dimension),
                nn.SELU()
            )
        

        self.moe=MoE(embedding_dimension,output_size,expert_size,embedding_dimension,model=MLP,k=k)
        
        
    def forward(self,des,tweet,num_prop,cat_prop,edge_index,edge_type):

        d=self.linear_relu_des(des)
        t=self.linear_relu_tweet(tweet)
        n=self.linear_relu_num_prop(num_prop)
        c=self.linear_relu_cat_prop(cat_prop)

            
        x=torch.cat((d,t,n,c),dim=1)
        
        
        x=self.linear_relu_input(x)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
       
        

        #x=self.linear_relu_input2(x)
        x=self.rgcn2(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn2(x,edge_index,edge_type)
        #residual connection
        #x+=torch.cat((d,t,n,c),dim=1)
        x=self.moe(x)
            
        return x
    
class BotRGCN_fmoe_soft(nn.Module):
    def __init__(self,input_size=0,output_size=0,hidden_size=0,des_size=768,tweet_size=768,num_prop_size=5,cat_prop_size=3,embedding_dimension=128,dropout=0.4,expert_size=2,k=1):
        super(BotRGCN_fmoe_soft, self).__init__()
        self.dropout = dropout
        self.embedding_dimension=embedding_dimension
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension/4)),
            nn.SELU()
        )

        self.linear_relu_tweet=nn.Sequential(
                nn.Linear(tweet_size,int(embedding_dimension/4)),
                nn.SELU()
            )
        self.linear_relu_num_prop=nn.Sequential(
                nn.Linear(num_prop_size,int(embedding_dimension/4)),
                nn.SELU()
            )
        self.linear_relu_cat_prop=nn.Sequential(
                nn.Linear(cat_prop_size,int(embedding_dimension/4)),
                nn.SELU()
            )
            
        self.linear_relu_input=nn.Sequential(
                nn.Linear(embedding_dimension,embedding_dimension),
                nn.SELU()
            )
            
        self.rgcn=RGCNConv(embedding_dimension,embedding_dimension,num_relations=2)
        

        self.moe=MoE(embedding_dimension,output_size,expert_size,embedding_dimension,model=MLP2,k=k)
        
        
    def forward(self,des,tweet,num_prop,cat_prop,edge_index,edge_type):

        d=self.linear_relu_des(des)
        t=self.linear_relu_tweet(tweet)
        n=self.linear_relu_num_prop(num_prop)
        c=self.linear_relu_cat_prop(cat_prop)

            
        x=torch.cat((d,t,n,c),dim=1)
        
        x=self.linear_relu_input(x)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        #residual connection
        #x+=torch.cat((d,t,n,c),dim=1)
        x=self.moe(x)
            
        return x

class BotRGCN_backbone(nn.Module):
    def __init__(self,input_size=0,output_size=0,hidden_size=0,des_size=768,tweet_size=768,num_prop_size=5,cat_prop_size=3,embedding_dimension=128,dropout=0.4):
        super(BotRGCN_backbone, self).__init__()
        self.dropout = dropout
        '''
        
        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(tweet_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(num_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(cat_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        '''
        self.linear_relu_input=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        
        
        self.rgcn=RGCNConv(embedding_dimension,embedding_dimension,num_relations=2)
        

        
        
        
    def forward(self,d,t,n,c,edge_index,edge_type):
        '''
        d=self.linear_relu_des(des)
        t=self.linear_relu_tweet(tweet)
        n=self.linear_relu_num_prop(num_prop)
        c=self.linear_relu_cat_prop(cat_prop)
        '''
        x=torch.cat((d,t,n,c),dim=1)
        
        x=self.linear_relu_input(x)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
            
        return x

class BotGAT_fmoe(nn.Module):
    def __init__(self,input_size=0,output_size=0,hidden_size=0,des_size=768,tweet_size=768,num_prop_size=5,cat_prop_size=3,embedding_dimension=128,dropout=0.4,expert_size=2,k=1):
        super(BotGAT_fmoe, self).__init__()
        self.dropout = dropout
        self.embedding_dimension=embedding_dimension
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension/4)),
            nn.SELU()
        )

        self.linear_relu_tweet=nn.Sequential(
                nn.Linear(tweet_size,int(embedding_dimension/4)),
                nn.SELU()
            )
        self.linear_relu_num_prop=nn.Sequential(
                nn.Linear(num_prop_size,int(embedding_dimension/4)),
                nn.SELU()
            )
        self.linear_relu_cat_prop=nn.Sequential(
                nn.Linear(cat_prop_size,int(embedding_dimension/4)),
                nn.SELU()
            )
            
        self.linear_relu_input=nn.Sequential(
                nn.Linear(embedding_dimension,embedding_dimension),
                nn.SELU()
            )
            
        self.gat1=GATConv(embedding_dimension,int(embedding_dimension/4),heads=4)
        self.gat2=GATConv(embedding_dimension,embedding_dimension)
        

        self.moe=MoE(embedding_dimension,output_size,expert_size,embedding_dimension,model=MLP,k=k)
        
        
    def forward(self,des,tweet,num_prop,cat_prop,edge_index,edge_type):

        d=self.linear_relu_des(des)
        t=self.linear_relu_tweet(tweet)
        n=self.linear_relu_num_prop(num_prop)
        c=self.linear_relu_cat_prop(cat_prop)

            
        x=torch.cat((d,t,n,c),dim=1)
        
        x=self.linear_relu_input(x)
        x=self.gat1(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.gat2(x,edge_index,edge_type)
        #residual connection
        #x+=torch.cat((d,t,n,c),dim=1)
        x=self.moe(x)
            
        return x

    
class BotGCN_fmoe(nn.Module):
    def __init__(self,input_size=0,output_size=0,hidden_size=0,des_size=768,tweet_size=768,num_prop_size=5,cat_prop_size=3,embedding_dimension=128,dropout=0.4,expert_size=2,k=1):
        super(BotGCN_fmoe, self).__init__()
        self.dropout = dropout
        self.embedding_dimension=embedding_dimension
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension/4)),
            nn.SELU()
        )

        self.linear_relu_tweet=nn.Sequential(
                nn.Linear(tweet_size,int(embedding_dimension/4)),
                nn.SELU()
            )
        self.linear_relu_num_prop=nn.Sequential(
                nn.Linear(num_prop_size,int(embedding_dimension/4)),
                nn.SELU()
            )
        self.linear_relu_cat_prop=nn.Sequential(
                nn.Linear(cat_prop_size,int(embedding_dimension/4)),
                nn.SELU()
            )
            
        self.linear_relu_input=nn.Sequential(
                nn.Linear(embedding_dimension,embedding_dimension),
                nn.SELU()
            )
            
        self.gcn1=GCNConv(embedding_dimension,embedding_dimension)
        self.gcn2=GCNConv(embedding_dimension,embedding_dimension)
        

        self.moe=MoE(embedding_dimension,output_size,expert_size,embedding_dimension,model=MLP,k=k)
        
        
    def forward(self,des,tweet,num_prop,cat_prop,edge_index,edge_type):

        d=self.linear_relu_des(des)
        t=self.linear_relu_tweet(tweet)
        n=self.linear_relu_num_prop(num_prop)
        c=self.linear_relu_cat_prop(cat_prop)

            
        x=torch.cat((d,t,n,c),dim=1)
        
        x=self.linear_relu_input(x)
        x=self.gcn1(x,edge_index)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.gcn2(x,edge_index)
        #residual connection
        #x+=torch.cat((d,t,n,c),dim=1)
        x=self.moe(x)
            
        return x

class BotRGAT_fmoe(nn.Module):
    def __init__(self,input_size=0,output_size=0,hidden_size=0,des_size=768,tweet_size=768,num_prop_size=5,cat_prop_size=3,embedding_dimension=128,dropout=0.4,expert_size=2,k=1):
        super(BotRGAT_fmoe, self).__init__()
        self.dropout = dropout
        self.embedding_dimension=embedding_dimension
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension/4)),
            nn.SELU()
        )

        self.linear_relu_tweet=nn.Sequential(
                nn.Linear(tweet_size,int(embedding_dimension/4)),
                nn.SELU()
            )
        self.linear_relu_num_prop=nn.Sequential(
                nn.Linear(num_prop_size,int(embedding_dimension/4)),
                nn.SELU()
            )
        self.linear_relu_cat_prop=nn.Sequential(
                nn.Linear(cat_prop_size,int(embedding_dimension/4)),
                nn.SELU()
            )
            
        self.linear_relu_input=nn.Sequential(
                nn.Linear(embedding_dimension,embedding_dimension),
                nn.SELU()
            )
            
        self.gcn1=RGATConv(embedding_dimension,int(embedding_dimension/4),heads=4,num_relations=2)
        #self.gcn2=RGATConv(embedding_dimension,int(embedding_dimension/4),heads=4,num_relations=2)
        

        self.moe=MoE(embedding_dimension,output_size,expert_size,embedding_dimension,model=MLP,k=k)
        
        
    def forward(self,des,tweet,num_prop,cat_prop,edge_index,edge_type):

        d=self.linear_relu_des(des)
        t=self.linear_relu_tweet(tweet)
        n=self.linear_relu_num_prop(num_prop)
        c=self.linear_relu_cat_prop(cat_prop)

            
        x=torch.cat((d,t,n,c),dim=1)
        
        x=self.linear_relu_input(x)
        x=self.gcn1(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.gcn1(x,edge_index,edge_type)
        #residual connection
        #x+=torch.cat((d,t,n,c),dim=1)
        x=self.moe(x)
            
        return x



def masked_edge_index(edge_index, edge_mask):
    return edge_index[:, edge_mask]

class SemanticAttention(torch.nn.Module):
    def __init__(self, in_channel, num_head, hidden_size=128):
        super(SemanticAttention, self).__init__()
        
        self.num_head = num_head
        self.att_layers = torch.nn.ModuleList()
        # multi-head attention
        for i in range(num_head):
            self.att_layers.append(
            torch.nn.Sequential(
                torch.nn.Linear(in_channel, hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden_size, 1, bias=False))
            )
       
    def forward(self, z):
        w = self.att_layers[0](z).mean(0)                    
        beta = torch.softmax(w, dim=0)                 
    
        beta = beta.expand((z.shape[0],) + beta.shape)
        output = (beta * z).sum(1)

        for i in range(1, self.num_head):
            w = self.att_layers[i](z).mean(0)
            beta = torch.softmax(w, dim=0)
            
            beta = beta.expand((z.shape[0],) + beta.shape)
            temp = (beta * z).sum(1)
            output += temp 
            
        return output / self.num_head

class RGTLayer(torch.nn.Module):
    def __init__(self, num_edge_type, in_channel, out_channel, trans_heads, semantic_head, dropout):
        super(RGTLayer, self).__init__()
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(in_channel + out_channel, in_channel),
            torch.nn.Sigmoid()
        )

        self.activation = torch.nn.ELU()
        self.transformer_list = torch.nn.ModuleList()
        for i in range(int(num_edge_type)):
            self.transformer_list.append(TransformerConv(in_channels=in_channel, out_channels=out_channel, heads=trans_heads, dropout=dropout, concat=False))
        
        self.num_edge_type = num_edge_type
        self.semantic_attention = SemanticAttention(in_channel=out_channel, num_head=semantic_head)

    def forward(self, features, edge_index, edge_type):
        r"""
        feature: input node features
        edge_index: all edge index, shape (2, num_edges)
        edge_type: same as RGCNconv in torch_geometric
        num_rel: number of relations
        beta: return cross relation attention weight
        agg: aggregation type across relation embedding
        """

        edge_index_list = []
        for i in range(self.num_edge_type):
            tmp = masked_edge_index(edge_index, edge_type == i)
            edge_index_list.append(tmp)

        u = self.transformer_list[0](features, edge_index_list[0].squeeze(0)).flatten(1) #.unsqueeze(1)
        a = self.gate(torch.cat((u, features), dim = 1))

        semantic_embeddings = (torch.mul(torch.tanh(u), a) + torch.mul(features, (1-a))).unsqueeze(1)
        
        for i in range(1,len(edge_index_list)):
            
            u = self.transformer_list[i](features, edge_index_list[i].squeeze(0)).flatten(1)
            a = self.gate(torch.cat((u, features), dim = 1))
            output = torch.mul(torch.tanh(u), a) + torch.mul(features, (1-a))
            semantic_embeddings=torch.cat((semantic_embeddings, output.unsqueeze(1)), dim = 1)
            
            return self.semantic_attention(semantic_embeddings)


class BotRGT_fmoe(nn.Module):
    def __init__(self,input_size=0,output_size=2,hidden_size=0,des_size=768,tweet_size=768,num_prop_size=5,cat_prop_size=3,embedding_dimension=128,dropout=0.4,expert_size=2,k=1):
        super(BotRGT_fmoe, self).__init__()
        self.dropout = dropout
        self.embedding_dimension=embedding_dimension
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension/4)),
            nn.SELU()
        )

        self.linear_relu_tweet=nn.Sequential(
                nn.Linear(tweet_size,int(embedding_dimension/4)),
                nn.SELU()
            )
        self.linear_relu_num_prop=nn.Sequential(
                nn.Linear(num_prop_size,int(embedding_dimension/4)),
                nn.SELU()
            )
        self.linear_relu_cat_prop=nn.Sequential(
                nn.Linear(cat_prop_size,int(embedding_dimension/4)),
                nn.SELU()
            )
            
        self.linear_relu_input=nn.Sequential(
                nn.Linear(embedding_dimension,embedding_dimension),
                nn.SELU()
            )
            
        self.gcn1=RGTLayer(2,embedding_dimension,embedding_dimension,2,2,0.5)
        self.gcn2=RGTLayer(2,embedding_dimension,embedding_dimension,2,2,0.5)
        

        self.moe=MoE(embedding_dimension,output_size,expert_size,64,model=MLP,k=k)
        
        
    def forward(self,des,tweet,num_prop,cat_prop,edge_index,edge_type):

        d=self.linear_relu_des(des)
        t=self.linear_relu_tweet(tweet)
        n=self.linear_relu_num_prop(num_prop)
        c=self.linear_relu_cat_prop(cat_prop)

            
        x=torch.cat((d,t,n,c),dim=1)
        
        x=self.linear_relu_input(x)
        x=self.gcn1(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.gcn2(x,edge_index,edge_type)
        #residual connection
        #x+=torch.cat((d,t,n,c),dim=1)
        x=self.moe(x)
            
        return x


class BotSimple_HGCN_fmoe(nn.Module):
    def __init__(self,input_size=0,output_size=0,hidden_size=0,des_size=768,tweet_size=768,num_prop_size=5,cat_prop_size=3,embedding_dimension=128,dropout=0.4,expert_size=2,k=1,beta=0.05,rel_dim=100):
    
        super(BotSimple_HGCN_fmoe, self).__init__()
        self.dropout = dropout
        self.embedding_dimension=embedding_dimension
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension/4)),
            nn.SELU()
        )

        self.linear_relu_tweet=nn.Sequential(
                nn.Linear(tweet_size,int(embedding_dimension/4)),
                nn.SELU()
            )
        self.linear_relu_num_prop=nn.Sequential(
                nn.Linear(num_prop_size,int(embedding_dimension/4)),
                nn.SELU()
            )
        self.linear_relu_cat_prop=nn.Sequential(
                nn.Linear(cat_prop_size,int(embedding_dimension/4)),
                nn.SELU()
            )
            
        self.linear_relu_input=nn.Sequential(
                nn.Linear(embedding_dimension,embedding_dimension),
                nn.SELU()
            )
            
        self.hgcn1=SimpleHGN(num_edge_type=2,in_channels=embedding_dimension,out_channels=embedding_dimension,rel_dim=rel_dim,beta=beta)
        self.hgcn2=SimpleHGN(num_edge_type=2,in_channels=embedding_dimension,out_channels=embedding_dimension,rel_dim=rel_dim,beta=beta,final_layer=True)

        self.moe=MoE(embedding_dimension,output_size,expert_size,embedding_dimension,model=MLP,k=k)
        
        
    def forward(self,des,tweet,num_prop,cat_prop,edge_index,edge_type):

        d=self.linear_relu_des(des)
        t=self.linear_relu_tweet(tweet)
        n=self.linear_relu_num_prop(num_prop)
        c=self.linear_relu_cat_prop(cat_prop)

            
        x=torch.cat((d,t,n,c),dim=1)
        
        x=self.linear_relu_input(x)
        x,alpha=self.hgcn1(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x,_=self.hgcn2(x,edge_index,edge_type,alpha)
        #residual connection
        #x+=torch.cat((d,t,n,c),dim=1)
        x=self.moe(x)
            
        return x