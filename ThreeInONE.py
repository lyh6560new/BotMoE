
import torch.nn as nn
from roberta import MLPclassifier
from Botrgcn import *
from moe_all import MoE,MoE_receive_gate
import torch
from Hayawi import DeeProBot_MoE,DeeProBot_MoE_gate,DeeProBot_MoE_bl
from  Transformer import LModel
import numpy as np


class feature_align(nn.Module):
    def __init__(self,align_size):
        super(feature_align,self).__init__()
        self.linear_relu_des=nn.Sequential(
            nn.Linear(768,int(align_size)),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(768,int(align_size)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(5,int(align_size)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(3,int(align_size)),
            nn.LeakyReLU()
        )
        self.linear_num_for_h_prop=nn.Sequential(
            nn.Linear(9,int(align_size)),
            nn.LeakyReLU()
        )
    def forward(self,des_tensor,tweets_tensor,num_prop,category_prop,num_for_h):
        des_tensor,tweets_tensor,num_prop,category_prop,num_for_h=self.linear_relu_des(des_tensor),self.linear_relu_tweet(tweets_tensor),\
                                                        self.linear_relu_num_prop(num_prop),self.linear_relu_cat_prop(category_prop),self.linear_num_for_h_prop(num_for_h)
        return des_tensor,tweets_tensor,num_prop,category_prop,num_for_h

class feature_align_new_fea(nn.Module):
    def __init__(self,align_size):
        super(feature_align_new_fea,self).__init__()
        self.linear_relu_des=nn.Sequential(
            nn.Linear(768,int(align_size)),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(768,int(align_size)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(27,int(align_size)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(11,int(align_size)),
            nn.LeakyReLU()
        )
        self.linear_num_for_h_prop=nn.Sequential(
            nn.Linear(9,int(align_size)),
            nn.LeakyReLU()
        )
    def forward(self,des_tensor,tweets_tensor,num_prop,category_prop,num_for_h):
        des_tensor,tweets_tensor,num_prop,category_prop,num_for_h=self.linear_relu_des(des_tensor),self.linear_relu_tweet(tweets_tensor),\
                                                        self.linear_relu_num_prop(num_prop),self.linear_relu_cat_prop(category_prop),self.linear_num_for_h_prop(num_for_h)
        return des_tensor,tweets_tensor,num_prop,category_prop,num_for_h

class feature_align1(nn.Module):
    def __init__(self,align_size):
        super(feature_align1,self).__init__()
        self.linear_relu_des=nn.Sequential(
            nn.Linear(768,int(align_size)),
            nn.LeakyReLU(),
            nn.Linear(int(align_size),int(align_size))
        )
        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(768,int(align_size)),
            nn.LeakyReLU(),
            nn.Linear(int(align_size),int(align_size))
        )
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(5,int(align_size)),
            nn.LeakyReLU(),
            nn.Linear(int(align_size),int(align_size))
        )
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(3,int(align_size)),
            nn.LeakyReLU(),
            nn.Linear(int(align_size),int(align_size))
        )
        self.linear_num_for_h_prop=nn.Sequential(
            nn.Linear(9,int(align_size)),
            nn.LeakyReLU(),
            nn.Linear(int(align_size),int(align_size))
        )
    def forward(self,des_tensor,tweets_tensor,num_prop,category_prop,num_for_h):
        des_tensor,tweets_tensor,num_prop,category_prop,num_for_h=self.linear_relu_des(des_tensor),self.linear_relu_tweet(tweets_tensor),\
                                                        self.linear_relu_num_prop(num_prop),self.linear_relu_cat_prop(category_prop),self.linear_num_for_h_prop(num_for_h)
        return des_tensor,tweets_tensor,num_prop,category_prop,num_for_h

class FixedPooling(nn.Module):
    def __init__(self, fixed_size):
        super().__init__()
        self.fixed_size = fixed_size

    def forward(self, x):
        b, w, h = x.shape
        p_w = self.fixed_size * ((w + self.fixed_size - 1) // self.fixed_size) - w
        p_h = self.fixed_size * ((h + self.fixed_size - 1) // self.fixed_size) - h
        x = nn.functional.pad(x, (0, p_h, 0, p_w))
        pool_size = (((w + self.fixed_size - 1) // self.fixed_size), ((h + self.fixed_size - 1) // self.fixed_size))
        pool = nn.MaxPool2d(pool_size, stride=pool_size)
        return pool(x)

class attentive_pooling(nn.Module):

    """ 点积注意力机制"""

    def __init__(self, attention_dropout=0.0):
        super(attentive_pooling, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """
        :param q:
        :param k:
        :param v:
        :param scale:
        :param attn_mask:
        :return: context tensor和attention tensor。
        """
        p=[q,k,v]
        for i in range(len(p)):
            if(p[i].dim()<3):
                p[i]=p[i].unsqueeze(1)
        q,k,v=p
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale        
        if attn_mask is not None:
            attention = attention.masked_fill(attn_mask, -np.inf)     
        # softmax
        attention = self.softmax(attention)
        # dropout
        attention = self.dropout(attention)
        # dot with v
        context = torch.bmm(attention, v)
        if(context.dim()==3):
            context=context.squeeze(1)
        if(attention.dim()==3):
            attention=attention.squeeze(1)

        return context, attention


class AllInOne(nn.Module):
#simple version
    def __init__(self, output_size,num_gnn,num_text,num_cat=2,kinds=2,gnn_k=1,text_k=1,cat_k=1,num_last_moe=4,align_size=32):
        super(AllInOne, self).__init__()
        self.output_size = output_size
        self.num_gnn=num_gnn
        self.num_text=num_text
        self.num_cat=num_cat
        
        self.align=feature_align(align_size=align_size)
        self.mlp_classifier=nn.Linear(output_size*3+6*6,2)
        self.fusion=LModel(embed_dim=output_size)
        self.gnn_moe=BotRGCN_fmoe1(align_size*4,self.output_size,128,expert_size=num_gnn,k=gnn_k,des_size=align_size,tweet_size=align_size,num_prop_size=align_size,cat_prop_size=align_size,embedding_dimension=align_size*4)
        self.text_moe=MoE(align_size,  self.output_size,num_text,128,model=MLPclassifier,k=text_k,all_mode=True )
        self.cat_moe=DeeProBot_MoE(num_prop_size=align_size,expert_size=num_cat,k=cat_k,output_size=output_size,moe_out=128)
        self.dropout=nn.Dropout(0.3)
        self.fixed_pooling=FixedPooling(fixed_size=6)
        self.bn1=nn.BatchNorm1d(output_size)
        self.bn2=nn.BatchNorm1d(output_size*3+6*6)
      
    
    def forward(self,des_tensor,tweets_tensor,num_prop,category_prop,num_for_h,edge_index,edge_type):

        
        des_tensor,tweets_tensor,num_prop,category_prop,num_for_h=self.align(des_tensor,tweets_tensor,num_prop,category_prop,num_for_h)

        gnn_out,exp_loss1=self.gnn_moe(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
        gnn_out=gnn_out[:11826]
        text_out,exp_loss2=self.text_moe.forward_roberta(tweets_tensor,des_tensor)
        text_out=text_out[:11826]
        cat_out,exp_loss3=self.cat_moe(num_for_h,category_prop[:11826])

        outputs=[gnn_out,text_out,cat_out]
        exp_loss=exp_loss1+exp_loss2+exp_loss3
        
        out_tensor=torch.stack(outputs,dim=1)
        out_tensor=self.bn1(out_tensor.permute(0,2,1)).permute(0,2,1)
        y=self.dropout(out_tensor)
        y,attention=self.fusion(out_tensor)
        attention=self.fixed_pooling(attention)

        
        y=torch.cat([y.reshape(len(y),self.output_size*3),attention.reshape(len(attention),-1)],dim=1)
        y=self.bn2(y)
        y=self.mlp_classifier(y)
        return y,exp_loss
    
class AllInOne1(nn.Module):
# feature align again in each model
    def __init__(self, output_size,num_gnn,num_text,num_cat=3,kinds=2,gnn_k=1,text_k=1,cat_k=1,num_last_moe=4,align_size=32):
        super(AllInOne1, self).__init__()
        self.output_size = output_size
        self.num_gnn=num_gnn
        self.num_text=num_text
        self.num_cat=num_cat        
        self.align=feature_align(align_size=align_size)
        self.mlp_classifier=nn.Linear(output_size*3+6*6,2)
        self.fusion=LModel(embed_dim=output_size)
        self.gnn_moe=BotRGCN_fmoe(align_size*4,self.output_size,128,expert_size=num_gnn,k=gnn_k,des_size=align_size,tweet_size=align_size,num_prop_size=align_size,cat_prop_size=align_size,embedding_dimension=align_size*4)
        self.text_moe=MoE(align_size,  self.output_size,num_text,128,model=MLPclassifier,k=text_k,all_mode=False )
        self.cat_moe=DeeProBot_MoE(num_prop_size=align_size,expert_size=num_cat,k=cat_k,output_size=output_size,moe_out=32)
        self.dropout=nn.Dropout(0.3)
        self.fixed_pooling=FixedPooling(fixed_size=6)
        self.bn1=nn.BatchNorm1d(output_size)
        self.bn2=nn.BatchNorm1d(output_size*3+6*6)
      
    
    def forward(self,des_tensor,tweets_tensor,num_prop,category_prop,num_for_h,edge_index,edge_type):
        des_tensor,tweets_tensor,num_prop,category_prop,num_for_h=self.align(des_tensor,tweets_tensor,num_prop,category_prop,num_for_h)
        gnn_out,exp_loss1=self.gnn_moe(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
        gnn_out=gnn_out[:11826]
        text_out,exp_loss2=self.text_moe.forward_roberta(tweets_tensor,des_tensor)
        text_out=text_out[:11826]
        cat_out,exp_loss3=self.cat_moe(num_for_h,category_prop[:11826])

        outputs=[gnn_out,text_out,cat_out]
        exp_loss=exp_loss1+exp_loss2+exp_loss3
        
        out_tensor=torch.stack(outputs,dim=1)
        out_tensor=self.bn1(out_tensor.permute(0,2,1)).permute(0,2,1)
        y=self.dropout(out_tensor)
        y,attention=self.fusion(out_tensor)
        attention=self.fixed_pooling(attention)
        
        y=torch.cat([y.reshape(len(y),self.output_size*3),attention.reshape(len(attention),-1)],dim=1)
        y=self.bn2(y)
        y=self.mlp_classifier(y)
        return y,exp_loss


class AllInOne2(nn.Module):

#augmented align
    def __init__(self, output_size,num_gnn,num_text,num_cat=3,kinds=2,gnn_k=1,text_k=1,cat_k=1,num_last_moe=4,align_size=32):
        super(AllInOne2, self).__init__()
        self.output_size = output_size
        self.num_gnn=num_gnn
        self.num_text=num_text
        self.num_cat=num_cat
        
        self.align=feature_align1(align_size=align_size)
        self.mlp_classifier=nn.Linear(output_size*3+6*6,2)
        self.fusion=LModel(embed_dim=output_size)
        self.gnn_moe=BotRGCN_fmoe1(align_size*4,self.output_size,128,expert_size=num_gnn,k=gnn_k,des_size=align_size,tweet_size=align_size,num_prop_size=align_size,cat_prop_size=align_size,embedding_dimension=align_size*4)
        self.text_moe=MoE(align_size,  self.output_size,num_text,128,model=MLPclassifier,k=text_k,all_mode=True )
        self.cat_moe=DeeProBot_MoE(num_prop_size=align_size,expert_size=num_cat,k=cat_k,output_size=output_size,moe_out=128)
        self.dropout=nn.Dropout(0.3)
        self.fixed_pooling=FixedPooling(fixed_size=6)
        self.bn1=nn.BatchNorm1d(output_size)
        self.bn2=nn.BatchNorm1d(output_size*3+6*6)
      
    
    def forward(self,des_tensor,tweets_tensor,num_prop,category_prop,num_for_h,edge_index,edge_type):

        
        des_tensor,tweets_tensor,num_prop,category_prop,num_for_h=self.align(des_tensor,tweets_tensor,num_prop,category_prop,num_for_h)

        gnn_out,exp_loss1=self.gnn_moe(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
        gnn_out=gnn_out[:11826]
        text_out,exp_loss2=self.text_moe.forward_roberta(tweets_tensor,des_tensor)
        text_out=text_out[:11826]
        cat_out,exp_loss3=self.cat_moe(num_for_h,category_prop[:11826])

        outputs=[gnn_out,text_out,cat_out]
        exp_loss=exp_loss1+exp_loss2+exp_loss3
        
        out_tensor=torch.stack(outputs,dim=1)
        out_tensor=self.bn1(out_tensor.permute(0,2,1)).permute(0,2,1)
        y=self.dropout(out_tensor)
        y,attention=self.fusion(out_tensor)
        attention=self.fixed_pooling(attention)

        
        y=torch.cat([y.reshape(len(y),self.output_size*3),attention.reshape(len(attention),-1)],dim=1)
        y=self.bn2(y)
        y=self.mlp_classifier(y)
        return y,exp_loss
    

class AllInOne1_soft(nn.Module):
# feature align again in each model
    def __init__(self, output_size,num_gnn,num_text,num_cat=3,kinds=2,gnn_k=1,text_k=1,cat_k=1,num_last_moe=4,align_size=32):
        super(AllInOne1_soft, self).__init__()
        self.output_size = output_size
        self.num_gnn=num_gnn
        self.num_text=num_text
        self.num_cat=num_cat        
        self.align=feature_align(align_size=align_size)
        self.mlp_classifier=nn.Linear(output_size*3+6*6,2)
        self.fusion=LModel(embed_dim=output_size)
        self.gnn_moe=BotRGCN_fmoe_soft(align_size*4,self.output_size,128,expert_size=num_gnn,k=gnn_k,des_size=align_size,tweet_size=align_size,num_prop_size=align_size,cat_prop_size=align_size,embedding_dimension=align_size*4)
        self.text_moe=MoE(align_size,  self.output_size,num_text,128,model=MLPclassifier,k=text_k,all_mode=False )
        self.cat_moe=DeeProBot_MoE(num_prop_size=align_size,expert_size=num_cat,k=cat_k,output_size=output_size,moe_out=32)
        self.dropout=nn.Dropout(0.3)
        self.fixed_pooling=FixedPooling(fixed_size=6)
        self.bn1=nn.BatchNorm1d(output_size)
        self.bn2=nn.BatchNorm1d(output_size*3+6*6)
      
    
    def forward(self,des_tensor,tweets_tensor,num_prop,category_prop,num_for_h,edge_index,edge_type):
        des_tensor,tweets_tensor,num_prop,category_prop,num_for_h=self.align(des_tensor,tweets_tensor,num_prop,category_prop,num_for_h)
        gnn_out,exp_loss1=self.gnn_moe(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
        gnn_out=gnn_out[:11826]
        text_out,exp_loss2=self.text_moe.forward_roberta(tweets_tensor,des_tensor)
        text_out=text_out[:11826]
        cat_out,exp_loss3=self.cat_moe(num_for_h,category_prop[:11826])

        outputs=[gnn_out,text_out,cat_out]
        exp_loss=exp_loss1+exp_loss2+exp_loss3
        
        out_tensor=torch.stack(outputs,dim=1)
        out_tensor=self.bn1(out_tensor.permute(0,2,1)).permute(0,2,1)
        y=self.dropout(out_tensor)
        y,attention=self.fusion(out_tensor)
        attention=self.fixed_pooling(attention)
        
        y=torch.cat([y.reshape(len(y),self.output_size*3),attention.reshape(len(attention),-1)],dim=1)
        y=self.bn2(y)
        y=self.mlp_classifier(y)
        return y,exp_loss



class AllInOne1_gate(nn.Module):
# feature align again in each model
    def __init__(self, output_size,num_gnn,num_text,num_cat=3,kinds=2,gnn_k=1,text_k=1,cat_k=1,num_last_moe=4,align_size=32):
        super(AllInOne1_gate, self).__init__()
        self.output_size = output_size
        self.num_gnn=num_gnn
        self.num_text=num_text
        self.num_cat=num_cat        
        self.align=feature_align(align_size=align_size)
        self.mlp_classifier=nn.Linear(output_size*3+6*6,2)
        self.fusion=LModel(embed_dim=output_size)
        self.gnn_moe=BotRGCN_fmoe_gate(align_size*4,self.output_size,128,expert_size=num_gnn,k=gnn_k,des_size=align_size,tweet_size=align_size,num_prop_size=align_size,cat_prop_size=align_size,embedding_dimension=align_size*4)
        self.text_moe=MoE_receive_gate(align_size,  self.output_size,num_text,128,model=MLPclassifier,k=text_k,all_mode=False )
        self.cat_moe=DeeProBot_MoE_gate(align_size*4,num_prop_size=align_size,expert_size=num_cat,k=cat_k,output_size=output_size,moe_out=32)
        self.dropout=nn.Dropout(0.3)
        self.fixed_pooling=FixedPooling(fixed_size=6)
        self.bn1=nn.BatchNorm1d(output_size)
        self.bn2=nn.BatchNorm1d(output_size*3+6*6)
      
    
    def forward(self,des_tensor,tweets_tensor,num_prop,category_prop,num_for_h,edge_index,edge_type):
        des_tensor,tweets_tensor,num_prop,category_prop,num_for_h=self.align(des_tensor,tweets_tensor,num_prop,category_prop,num_for_h)
        gnn_out,exp_loss1,gate=self.gnn_moe(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
        gnn_out=gnn_out[:11826]
        text_out=self.text_moe.forward_roberta(tweets_tensor,des_tensor,gate)
        text_out=text_out[:11826]
        cat_out=self.cat_moe(num_for_h,category_prop[:11826],gate)

        outputs=[gnn_out,text_out,cat_out]
        exp_loss=exp_loss1
        
        out_tensor=torch.stack(outputs,dim=1)
        out_tensor=self.bn1(out_tensor.permute(0,2,1)).permute(0,2,1)
        y=self.dropout(out_tensor)
        y,attention=self.fusion(out_tensor)
        attention=self.fixed_pooling(attention)
        
        y=torch.cat([y.reshape(len(y),self.output_size*3),attention.reshape(len(attention),-1)],dim=1)
        y=self.bn2(y)
        y=self.mlp_classifier(y)
        return y,exp_loss

class AllInOne1_individual_gate(nn.Module):
# feature align again in each model
    def __init__(self, output_size,num_gnn,num_text,num_cat=3,kinds=2,gnn_k=1,text_k=1,cat_k=1,num_last_moe=4,align_size=32):
        super(AllInOne1_individual_gate, self).__init__()
        self.output_size = output_size
        self.num_gnn=num_gnn
        self.num_text=num_text
        self.num_cat=num_cat        
        self.align=feature_align(align_size=align_size)
        self.mlp_classifier=nn.Linear(output_size*3+6*6,2)
        self.fusion=LModel(embed_dim=output_size)
        self.gnn_moe=BotRGCN_fmoe_gate(align_size*4,self.output_size,128,expert_size=num_gnn,k=gnn_k,des_size=align_size,tweet_size=align_size,num_prop_size=align_size,cat_prop_size=align_size,embedding_dimension=align_size*4)
        self.text_gnn=BotRGCN_backbone(align_size*4,self.output_size,128,des_size=align_size,tweet_size=align_size,num_prop_size=align_size,cat_prop_size=align_size,embedding_dimension=align_size*4)
        self.text_moe=MoE_receive_gate(align_size,  self.output_size,num_text,128,model=MLPclassifier,k=text_k,all_mode=False,special_for_gate=align_size*4)
        self.cat_gnn=BotRGCN_backbone(align_size*4,self.output_size,128,des_size=align_size,tweet_size=align_size,num_prop_size=align_size,cat_prop_size=align_size,embedding_dimension=align_size*4)
        self.cat_moe=DeeProBot_MoE_gate(align_size*4,num_prop_size=align_size,expert_size=num_cat,k=cat_k,output_size=output_size,moe_out=32)
        self.dropout=nn.Dropout(0.3)
        self.fixed_pooling=FixedPooling(fixed_size=6)
        self.bn1=nn.BatchNorm1d(output_size)
        self.bn2=nn.BatchNorm1d(output_size*3+6*6)
      
    
    def forward(self,des_tensor,tweets_tensor,num_prop,category_prop,num_for_h,edge_index,edge_type):
        des_tensor,tweets_tensor,num_prop,category_prop,num_for_h=self.align(des_tensor,tweets_tensor,num_prop,category_prop,num_for_h)
        gnn_out,exp_loss1,gate=self.gnn_moe(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
        gate_text_fea=self.text_gnn(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
        #gate_cat_fea=self.cat_gnn(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
        gate_cat_fea=gate_text_fea
        gnn_out=gnn_out[:11826]
        text_out,exp_loss2=self.text_moe.forward_roberta(tweets_tensor,des_tensor,gate_text_fea)
        text_out=text_out[:11826]
        cat_out,exp_loss3=self.cat_moe(num_for_h,category_prop[:11826],gate_cat_fea)

        outputs=[gnn_out,text_out,cat_out]
        exp_loss=exp_loss1+exp_loss2+exp_loss3
        
        out_tensor=torch.stack(outputs,dim=1)
        out_tensor=self.bn1(out_tensor.permute(0,2,1)).permute(0,2,1)
        y=self.dropout(out_tensor)
        y,attention=self.fusion(out_tensor)
        attention=self.fixed_pooling(attention)
        
        y=torch.cat([y.reshape(len(y),self.output_size*3),attention.reshape(len(attention),-1)],dim=1)
        y=self.bn2(y)
        y=self.mlp_classifier(y)
        return y,exp_loss

class AllInOne1_RGT(nn.Module):
# feature align again in each model
    def __init__(self, output_size,num_gnn,num_text,num_cat=3,kinds=2,gnn_k=1,text_k=1,cat_k=1,num_last_moe=4,align_size=32):
        super(AllInOne1_RGT, self).__init__()
        self.output_size = output_size
        self.num_gnn=num_gnn
        self.num_text=num_text
        self.num_cat=num_cat        
        self.align=feature_align(align_size=align_size)
        self.mlp_classifier=nn.Linear(output_size*3+6*6,2)
        self.fusion=LModel(embed_dim=output_size)
        self.gnn_moe=BotRGT_fmoe(align_size*4,self.output_size,128,expert_size=num_gnn,k=gnn_k,des_size=align_size,tweet_size=align_size,num_prop_size=align_size,cat_prop_size=align_size,embedding_dimension=align_size*4)
        self.text_moe=MoE(align_size,  self.output_size,num_text,128,model=MLPclassifier,k=text_k,all_mode=False )
        self.cat_moe=DeeProBot_MoE(num_prop_size=align_size,expert_size=num_cat,k=cat_k,output_size=output_size,moe_out=32)
        self.dropout=nn.Dropout(0.3)
        self.fixed_pooling=FixedPooling(fixed_size=6)
        self.bn1=nn.BatchNorm1d(output_size)
        self.bn2=nn.BatchNorm1d(output_size*3+6*6)
      
    
    def forward(self,des_tensor,tweets_tensor,num_prop,category_prop,num_for_h,edge_index,edge_type):
        des_tensor,tweets_tensor,num_prop,category_prop,num_for_h=self.align(des_tensor,tweets_tensor,num_prop,category_prop,num_for_h)
        gnn_out,exp_loss1=self.gnn_moe(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
        gnn_out=gnn_out[:11826]
        text_out,exp_loss2=self.text_moe.forward_roberta(tweets_tensor,des_tensor)
        text_out=text_out[:11826]
        cat_out,exp_loss3=self.cat_moe(num_for_h,category_prop[:11826])

        outputs=[gnn_out,text_out,cat_out]
        exp_loss=exp_loss1+exp_loss2+exp_loss3
        
        out_tensor=torch.stack(outputs,dim=1)
        out_tensor=self.bn1(out_tensor.permute(0,2,1)).permute(0,2,1)
        y=self.dropout(out_tensor)
        y,attention=self.fusion(out_tensor)
        attention=self.fixed_pooling(attention)
        
        y=torch.cat([y.reshape(len(y),self.output_size*3),attention.reshape(len(attention),-1)],dim=1)
        y=self.bn2(y)
        y=self.mlp_classifier(y)
        return y,exp_loss

class AllInOne1_gcn(nn.Module):
# feature align again in each model
    def __init__(self, output_size,num_gnn,num_text,num_cat=3,kinds=2,gnn_k=1,text_k=1,cat_k=1,num_last_moe=4,align_size=32):
        super(AllInOne1_gcn, self).__init__()
        self.output_size = output_size
        self.num_gnn=num_gnn
        self.num_text=num_text
        self.num_cat=num_cat        
        self.align=feature_align(align_size=align_size)
        self.mlp_classifier=nn.Linear(output_size*3+6*6,2)
        self.fusion=LModel(embed_dim=output_size)
        self.gnn_moe=BotGCN_fmoe(align_size*4,self.output_size,128,expert_size=num_gnn,k=gnn_k,des_size=align_size,tweet_size=align_size,num_prop_size=align_size,cat_prop_size=align_size,embedding_dimension=align_size*4)
        self.text_moe=MoE(align_size,  self.output_size,num_text,128,model=MLPclassifier,k=text_k,all_mode=False )
        self.cat_moe=DeeProBot_MoE(num_prop_size=align_size,expert_size=num_cat,k=cat_k,output_size=output_size,moe_out=32)
        self.dropout=nn.Dropout(0.3)
        self.fixed_pooling=FixedPooling(fixed_size=6)
        self.bn1=nn.BatchNorm1d(output_size)
        self.bn2=nn.BatchNorm1d(output_size*3+6*6)
      
    
    def forward(self,des_tensor,tweets_tensor,num_prop,category_prop,num_for_h,edge_index,edge_type):
        des_tensor,tweets_tensor,num_prop,category_prop,num_for_h=self.align(des_tensor,tweets_tensor,num_prop,category_prop,num_for_h)
        gnn_out,exp_loss1=self.gnn_moe(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
        gnn_out=gnn_out[:11826]
        text_out,exp_loss2=self.text_moe.forward_roberta(tweets_tensor,des_tensor)
        text_out=text_out[:11826]
        cat_out,exp_loss3=self.cat_moe(num_for_h,category_prop[:11826])

        outputs=[gnn_out,text_out,cat_out]
        exp_loss=exp_loss1+exp_loss2+exp_loss3
        
        out_tensor=torch.stack(outputs,dim=1)
        out_tensor=self.bn1(out_tensor.permute(0,2,1)).permute(0,2,1)
        y=self.dropout(out_tensor)
        y,attention=self.fusion(out_tensor)
        attention=self.fixed_pooling(attention)
        
        y=torch.cat([y.reshape(len(y),self.output_size*3),attention.reshape(len(attention),-1)],dim=1)
        y=self.bn2(y)
        y=self.mlp_classifier(y)
        return y,exp_loss


class AllInOne1_rgat(nn.Module):
# feature align again in each model
    def __init__(self, output_size,num_gnn,num_text,num_cat=3,kinds=2,gnn_k=1,text_k=1,cat_k=1,num_last_moe=4,align_size=32):
        super(AllInOne1_rgat, self).__init__()
        self.output_size = output_size
        self.num_gnn=num_gnn
        self.num_text=num_text
        self.num_cat=num_cat        
        self.align=feature_align(align_size=align_size)
        self.mlp_classifier=nn.Linear(output_size*3+6*6,2)
        self.fusion=LModel(embed_dim=output_size)
        self.gnn_moe=BotRGAT_fmoe(align_size*4,self.output_size,128,expert_size=num_gnn,k=gnn_k,des_size=align_size,tweet_size=align_size,num_prop_size=align_size,cat_prop_size=align_size,embedding_dimension=align_size*4)
        self.text_moe=MoE(align_size,  self.output_size,num_text,128,model=MLPclassifier,k=text_k,all_mode=False )
        self.cat_moe=DeeProBot_MoE(num_prop_size=align_size,expert_size=num_cat,k=cat_k,output_size=output_size,moe_out=32)
        self.dropout=nn.Dropout(0.3)
        self.fixed_pooling=FixedPooling(fixed_size=6)
        self.bn1=nn.BatchNorm1d(output_size)
        self.bn2=nn.BatchNorm1d(output_size*3+6*6)
      
    
    def forward(self,des_tensor,tweets_tensor,num_prop,category_prop,num_for_h,edge_index,edge_type):
        des_tensor,tweets_tensor,num_prop,category_prop,num_for_h=self.align(des_tensor,tweets_tensor,num_prop,category_prop,num_for_h)
        gnn_out,exp_loss1=self.gnn_moe(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
        gnn_out=gnn_out[:11826]
        text_out,exp_loss2=self.text_moe.forward_roberta(tweets_tensor,des_tensor)
        text_out=text_out[:11826]
        cat_out,exp_loss3=self.cat_moe(num_for_h,category_prop[:11826])

        outputs=[gnn_out,text_out,cat_out]
        exp_loss=exp_loss1+exp_loss2+exp_loss3
        
        out_tensor=torch.stack(outputs,dim=1)
        out_tensor=self.bn1(out_tensor.permute(0,2,1)).permute(0,2,1)
        y=self.dropout(out_tensor)
        y,attention=self.fusion(out_tensor)
        attention=self.fixed_pooling(attention)
        
        y=torch.cat([y.reshape(len(y),self.output_size*3),attention.reshape(len(attention),-1)],dim=1)
        y=self.bn2(y)
        y=self.mlp_classifier(y)
        return y,exp_loss


class AllInOne1_rgcn_rgt_gcn(nn.Module):
# feature align again in each model
    def __init__(self, output_size,num_gnn,num_text,num_cat=3,kinds=2,gnn_k=1,text_k=1,cat_k=1,num_last_moe=4,align_size=32):
        super(AllInOne1_rgcn_rgt_gcn, self).__init__()
        self.output_size = output_size
        self.num_gnn=num_gnn
        self.num_text=num_text
        self.num_cat=num_cat        
        self.align=feature_align(align_size=align_size)
        self.mlp_classifier=nn.Linear(output_size*5+6*6,2)
        self.fusion=LModel(embed_dim=output_size)
        self.rgcn_moe=BotRGCN_fmoe(align_size*4,self.output_size,128,expert_size=num_gnn,k=gnn_k,des_size=align_size,tweet_size=align_size,num_prop_size=align_size,cat_prop_size=align_size,embedding_dimension=align_size*4)
        self.rgt_moe=BotRGT_fmoe(align_size*4,self.output_size,128,expert_size=num_gnn,k=gnn_k,des_size=align_size,tweet_size=align_size,num_prop_size=align_size,cat_prop_size=align_size,embedding_dimension=align_size*4)
        self.gcn_moe=BotGCN_fmoe(align_size*4,self.output_size,128,expert_size=num_gnn,k=gnn_k,des_size=align_size,tweet_size=align_size,num_prop_size=align_size,cat_prop_size=align_size,embedding_dimension=align_size*4)
        self.text_moe=MoE(align_size,  self.output_size,num_text,128,model=MLPclassifier,k=text_k,all_mode=False )
        self.cat_moe=DeeProBot_MoE(num_prop_size=align_size,expert_size=num_cat,k=cat_k,output_size=output_size,moe_out=32)
        self.dropout=nn.Dropout(0.3)
        self.fixed_pooling=FixedPooling(fixed_size=6)
        self.bn1=nn.BatchNorm1d(output_size)
        self.bn2=nn.BatchNorm1d(output_size*5+6*6)
      
    
    def forward(self,des_tensor,tweets_tensor,num_prop,category_prop,num_for_h,edge_index,edge_type):
        des_tensor,tweets_tensor,num_prop,category_prop,num_for_h=self.align(des_tensor,tweets_tensor,num_prop,category_prop,num_for_h)
        gcn_out,exp_loss1=self.gcn_moe(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
        gcn_out=gcn_out[:11826]
        rgcn_out,exp_loss2=self.rgcn_moe(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
        rgcn_out=rgcn_out[:11826]
        rgt_out,exp_loss3=self.rgt_moe(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
        rgt_out=rgt_out[:11826]
        text_out,exp_loss4=self.text_moe.forward_roberta(tweets_tensor,des_tensor)
        text_out=text_out[:11826]
        cat_out,exp_loss5=self.cat_moe(num_for_h,category_prop[:11826])

        outputs=[gcn_out,rgcn_out,rgt_out,text_out,cat_out]
        exp_loss=exp_loss1+exp_loss2+exp_loss3+exp_loss4+exp_loss5
        
        out_tensor=torch.stack(outputs,dim=1)
        out_tensor=self.bn1(out_tensor.permute(0,2,1)).permute(0,2,1)
        y=self.dropout(out_tensor)
        y,attention=self.fusion(out_tensor)
        attention=self.fixed_pooling(attention)
        
        y=torch.cat([y.reshape(len(y),self.output_size*5),attention.reshape(len(attention),-1)],dim=1)
        y=self.bn2(y)
        y=self.mlp_classifier(y)
        return y,exp_loss

class feature_align_with_t5(nn.Module):
    def __init__(self,align_size):
        super(feature_align_with_t5,self).__init__()
        self.linear_relu_des=nn.Sequential(
            nn.Linear(768,int(align_size)),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(768,int(align_size)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(5,int(align_size)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(3,int(align_size)),
            nn.LeakyReLU()
        )
        self.linear_num_for_h_prop=nn.Sequential(
            nn.Linear(9,int(align_size)),
            nn.LeakyReLU()
        )
        self.linear_relu_des_t5=nn.Sequential(
            nn.Linear(512,int(align_size)),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet_t5=nn.Sequential(
            nn.Linear(512,int(align_size)),
            nn.LeakyReLU()
        )
    def forward(self,des_tensor,tweets_tensor,num_prop,category_prop,num_for_h,des_t5,tweet_t5):
        des_tensor,tweets_tensor,num_prop,category_prop,num_for_h,des_t5,tweet_t5=self.linear_relu_des(des_tensor),self.linear_relu_tweet(tweets_tensor),\
                                                        self.linear_relu_num_prop(num_prop),self.linear_relu_cat_prop(category_prop),\
                                                        self.linear_num_for_h_prop(num_for_h),self.linear_relu_des_t5(des_t5),\
                                                        self.linear_relu_des_t5(tweet_t5)
        return des_tensor,tweets_tensor,num_prop,category_prop,num_for_h,des_t5,tweet_t5

class AllInOne1_rgcn_rgt_gcn_t5(nn.Module):
# feature align again in each model
    def __init__(self, output_size,num_gnn,num_text,num_cat=3,kinds=2,gnn_k=1,text_k=1,cat_k=1,num_last_moe=4,align_size=32):
        super(AllInOne1_rgcn_rgt_gcn_t5, self).__init__()
        self.output_size = output_size
        self.num_gnn=num_gnn
        self.num_text=num_text
        self.num_cat=num_cat        
        self.align=feature_align_with_t5(align_size=align_size)
        self.mlp_classifier=nn.Linear(output_size*6+6*6,2)
        self.fusion=LModel(embed_dim=output_size)
        self.rgcn_moe=BotRGCN_fmoe(align_size*4,self.output_size,128,expert_size=num_gnn,k=gnn_k,des_size=align_size,tweet_size=align_size,num_prop_size=align_size,cat_prop_size=align_size,embedding_dimension=align_size*4)
        self.rgt_moe=BotRGT_fmoe(align_size*4,self.output_size,128,expert_size=num_gnn,k=gnn_k,des_size=align_size,tweet_size=align_size,num_prop_size=align_size,cat_prop_size=align_size,embedding_dimension=align_size*4)
        self.gcn_moe=BotGCN_fmoe(align_size*4,self.output_size,128,expert_size=num_gnn,k=gnn_k,des_size=align_size,tweet_size=align_size,num_prop_size=align_size,cat_prop_size=align_size,embedding_dimension=align_size*4)
        self.text_moe=MoE(align_size,  self.output_size,num_text,128,model=MLPclassifier,k=text_k,all_mode=False )
        self.text_moe_t5=MoE(align_size,  self.output_size,num_text,128,model=MLPclassifier,k=text_k,all_mode=False )
        self.cat_moe=DeeProBot_MoE(num_prop_size=align_size,expert_size=num_cat,k=cat_k,output_size=output_size,moe_out=32)
        self.dropout=nn.Dropout(0.3)
        self.fixed_pooling=FixedPooling(fixed_size=6)
        self.bn1=nn.BatchNorm1d(output_size)
        self.bn2=nn.BatchNorm1d(output_size*6+6*6)
      
    
    def forward(self,des_tensor,tweets_tensor,num_prop,category_prop,num_for_h,des_t5,tweets_t5,edge_index,edge_type):
        des_tensor,tweets_tensor,num_prop,category_prop,num_for_h,des_t5,tweets_t5=self.align(des_tensor,tweets_tensor,num_prop,category_prop,num_for_h,des_t5,tweets_t5)
        gcn_out,exp_loss1=self.gcn_moe(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
        gcn_out=gcn_out[:11826]
        rgcn_out,exp_loss2=self.rgcn_moe(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
        rgcn_out=rgcn_out[:11826]
        rgt_out,exp_loss3=self.rgt_moe(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
        rgt_out=rgt_out[:11826]
        text_out,exp_loss4=self.text_moe.forward_roberta(tweets_tensor,des_tensor)
        text_out=text_out[:11826]
        text_out2,exp_loss5=self.text_moe_t5.forward_roberta(tweets_t5,des_t5)
        text_out2=text_out[:11826]
        cat_out,exp_loss6=self.cat_moe(num_for_h,category_prop[:11826])

        outputs=[gcn_out,rgcn_out,rgt_out,text_out,text_out2,cat_out]
        exp_loss=exp_loss1+exp_loss2+exp_loss3+exp_loss4+exp_loss5+exp_loss6
        
        out_tensor=torch.stack(outputs,dim=1)
        out_tensor=self.bn1(out_tensor.permute(0,2,1)).permute(0,2,1)
        y=self.dropout(out_tensor)
        y,attention=self.fusion(out_tensor)
        attention=self.fixed_pooling(attention)
        
        y=torch.cat([y.reshape(len(y),self.output_size*6),attention.reshape(len(attention),-1)],dim=1)
        y=self.bn2(y)
        y=self.mlp_classifier(y)
        return y,exp_loss

class AllInOne1_rgcn_rgt_gcn_simple_HGCN(nn.Module):
    def __init__(self, output_size,num_gnn,num_text,num_cat=3,kinds=2,gnn_k=1,text_k=1,cat_k=1,num_last_moe=4,align_size=32):
        super(AllInOne1_rgcn_rgt_gcn_simple_HGCN, self).__init__()
        self.output_size = output_size
        self.num_gnn=num_gnn
        self.num_text=num_text
        self.num_cat=num_cat        
        self.align=feature_align(align_size=align_size)
        self.mlp_classifier=nn.Linear(output_size*6+6*6,2)
        self.fusion=LModel(embed_dim=output_size)
        self.rgcn_moe=BotRGCN_fmoe(align_size*4,self.output_size,128,expert_size=num_gnn,k=gnn_k,des_size=align_size,tweet_size=align_size,num_prop_size=align_size,cat_prop_size=align_size,embedding_dimension=align_size*4)
        self.rgt_moe=BotRGT_fmoe(align_size*4,self.output_size,128,expert_size=num_gnn,k=gnn_k,des_size=align_size,tweet_size=align_size,num_prop_size=align_size,cat_prop_size=align_size,embedding_dimension=align_size*4)
        self.gcn_moe=BotGCN_fmoe(align_size*4,self.output_size,128,expert_size=num_gnn,k=gnn_k,des_size=align_size,tweet_size=align_size,num_prop_size=align_size,cat_prop_size=align_size,embedding_dimension=align_size*4)
        self.shgcn_moe=BotSimple_HGCN_fmoe(align_size*4,self.output_size,128,expert_size=num_gnn,k=gnn_k,des_size=align_size,tweet_size=align_size,num_prop_size=align_size,cat_prop_size=align_size,embedding_dimension=align_size*4)
        self.text_moe=MoE(align_size,  self.output_size,num_text,128,model=MLPclassifier,k=text_k,all_mode=False )
        self.cat_moe=DeeProBot_MoE(num_prop_size=align_size,expert_size=num_cat,k=cat_k,output_size=output_size,moe_out=32)
        self.dropout=nn.Dropout(0.3)
        self.fixed_pooling=FixedPooling(fixed_size=6)
        self.bn1=nn.BatchNorm1d(output_size)
        self.bn2=nn.BatchNorm1d(output_size*6+6*6)
      
    
    def forward(self,des_tensor,tweets_tensor,num_prop,category_prop,num_for_h,edge_index,edge_type):
        des_tensor,tweets_tensor,num_prop,category_prop,num_for_h=self.align(des_tensor,tweets_tensor,num_prop,category_prop,num_for_h)
        gcn_out,exp_loss1=self.gcn_moe(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
        gcn_out=gcn_out[:11826]
        rgcn_out,exp_loss2=self.rgcn_moe(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
        rgcn_out=rgcn_out[:11826]
        rgt_out,exp_loss3=self.rgt_moe(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
        rgt_out=rgt_out[:11826]
        shgcn_out,exp_loss4=self.shgcn_moe(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
        shgcn_out=shgcn_out[:11826]
        text_out,exp_loss4=self.text_moe.forward_roberta(tweets_tensor,des_tensor)
        text_out=text_out[:11826]
        cat_out,exp_loss5=self.cat_moe(num_for_h,category_prop[:11826])

        outputs=[gcn_out,rgcn_out,rgt_out,shgcn_out,text_out,cat_out]
        exp_loss=exp_loss1+exp_loss2+exp_loss3+exp_loss4+exp_loss5
        
        out_tensor=torch.stack(outputs,dim=1)
        out_tensor=self.bn1(out_tensor.permute(0,2,1)).permute(0,2,1)
        y=self.dropout(out_tensor)
        y,attention=self.fusion(out_tensor)
        attention=self.fixed_pooling(attention)
        
        y=torch.cat([y.reshape(len(y),self.output_size*6),attention.reshape(len(attention),-1)],dim=1)
        y=self.bn2(y)
        y=self.mlp_classifier(y)
        return y,exp_loss




class AllInOne1_new_fea(nn.Module):
# feature align again in each model
    def __init__(self, output_size,num_gnn,num_text,num_cat=3,kinds=2,gnn_k=1,text_k=1,cat_k=1,num_last_moe=4,align_size=32):
        super(AllInOne1_new_fea, self).__init__()
        self.output_size = output_size
        self.num_gnn=num_gnn
        self.num_text=num_text
        self.num_cat=num_cat        
        self.align=feature_align_new_fea(align_size=align_size)
        self.mlp_classifier=nn.Linear(output_size*3+6*6,2)
        self.fusion=LModel(embed_dim=output_size)
        self.gnn_moe=BotRGCN_fmoe(align_size*4,self.output_size,128,expert_size=num_gnn,k=gnn_k,des_size=align_size,tweet_size=align_size,num_prop_size=align_size,cat_prop_size=align_size,embedding_dimension=align_size*4)
        self.text_moe=MoE(align_size,  self.output_size,num_text,128,model=MLPclassifier,k=text_k,all_mode=False )
        self.cat_moe=DeeProBot_MoE(num_prop_size=align_size,expert_size=num_cat,k=cat_k,output_size=output_size,moe_out=32)
        self.dropout=nn.Dropout(0.3)
        self.fixed_pooling=FixedPooling(fixed_size=6)
        self.bn1=nn.BatchNorm1d(output_size)
        self.bn2=nn.BatchNorm1d(output_size*3+6*6)
      
    
    def forward(self,des_tensor,tweets_tensor,num_prop,category_prop,num_for_h,edge_index,edge_type):
        des_tensor,tweets_tensor,num_prop,category_prop,num_for_h=self.align(des_tensor,tweets_tensor,num_prop,category_prop,num_for_h)
        gnn_out,exp_loss1=self.gnn_moe(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
        text_out,exp_loss2=self.text_moe.forward_roberta(tweets_tensor,des_tensor)
        cat_out,exp_loss3=self.cat_moe(num_for_h,category_prop)

        outputs=[gnn_out,text_out,cat_out]
        exp_loss=exp_loss1+exp_loss2+exp_loss3
        
        out_tensor=torch.stack(outputs,dim=1)
        out_tensor=self.bn1(out_tensor.permute(0,2,1)).permute(0,2,1)
        y=self.dropout(out_tensor)
        y,attention=self.fusion(out_tensor)
        attention=self.fixed_pooling(attention)
        
        y=torch.cat([y.reshape(len(y),self.output_size*3),attention.reshape(len(attention),-1)],dim=1)
        y=self.bn2(y)
        y=self.mlp_classifier(y)
        return y,exp_loss

class AllInOne3(nn.Module):
# feature align again in each model
    def __init__(self, output_size,num_gnn,num_text,num_cat=3,kinds=2,gnn_k=1,text_k=1,cat_k=1,num_last_moe=4,align_size=32):
        super(AllInOne3, self).__init__()
        self.output_size = output_size
        self.num_gnn=num_gnn
        self.num_text=num_text
        self.num_cat=num_cat        
        self.align=feature_align(align_size=align_size)
        self.mlp_classifier=nn.Linear(output_size*3+2*2,2)
        self.fusion=LModel(embed_dim=output_size)
        self.gnn_moe=BotRGCN_fmoe(align_size*4,self.output_size,128,expert_size=num_gnn,k=gnn_k,des_size=align_size,tweet_size=align_size,num_prop_size=align_size,cat_prop_size=align_size,embedding_dimension=align_size*4)
        self.text_moe=MoE(align_size,  self.output_size,num_text,128,model=MLPclassifier,k=text_k,all_mode=False )
        self.cat_moe=DeeProBot_MoE(num_prop_size=align_size,expert_size=num_cat,k=cat_k,output_size=output_size,moe_out=32)
        self.dropout=nn.Dropout(0.3)
        self.fixed_pooling=FixedPooling(fixed_size=2)
        self.bn1=nn.BatchNorm1d(output_size)
        self.bn2=nn.BatchNorm1d(output_size*3+2*2)
      
    
    def forward(self,des_tensor,tweets_tensor,num_prop,category_prop,num_for_h,edge_index,edge_type):
        des_tensor,tweets_tensor,num_prop,category_prop,num_for_h=self.align(des_tensor,tweets_tensor,num_prop,category_prop,num_for_h)
        gnn_out,exp_loss1=self.gnn_moe(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
        gnn_out=gnn_out[:11826]
        text_out,exp_loss2=self.text_moe.forward_roberta(tweets_tensor,des_tensor)
        text_out=text_out[:11826]
        cat_out,exp_loss3=self.cat_moe(num_for_h,category_prop[:11826])

        outputs=[gnn_out,text_out,cat_out]
        exp_loss=exp_loss1+exp_loss2+exp_loss3
        
        out_tensor=torch.stack(outputs,dim=1)
        out_tensor=self.bn1(out_tensor.permute(0,2,1)).permute(0,2,1)
        y=self.dropout(out_tensor)
        y,attention=self.fusion(out_tensor)
        attention=self.fixed_pooling(attention)
        
        y=torch.cat([y.reshape(len(y),self.output_size*3),attention.reshape(len(attention),-1)],dim=1)
        y=self.bn2(y)
        y=self.mlp_classifier(y)
        return y,exp_loss



class AllInOne1_bl(nn.Module):
# feature align again in each model
    def __init__(self, output_size,num_gnn,num_text,num_cat=3,kinds=2,gnn_k=1,text_k=1,cat_k=1,num_last_moe=4,align_size=32,bl_cat=1e-2):
        super(AllInOne1_bl, self).__init__()
        self.output_size = output_size
        self.num_gnn=num_gnn
        self.num_text=num_text
        self.num_cat=num_cat        
        self.align=feature_align(align_size=align_size)
        self.mlp_classifier=nn.Linear(output_size*3+6*6,2)
        self.fusion=LModel(embed_dim=output_size)
        self.gnn_moe=BotRGCN_fmoe(align_size*4,self.output_size,128,expert_size=num_gnn,k=gnn_k,des_size=align_size,tweet_size=align_size,num_prop_size=align_size,cat_prop_size=align_size,embedding_dimension=align_size*4)
        self.text_moe=MoE(align_size,  self.output_size,num_text,128,model=MLPclassifier,k=text_k,all_mode=False )
        self.cat_moe=DeeProBot_MoE_bl(num_prop_size=align_size,expert_size=num_cat,k=cat_k,output_size=output_size,moe_out=32,loss_cof=bl_cat)
        self.dropout=nn.Dropout(0.3)
        self.fixed_pooling=FixedPooling(fixed_size=6)
        self.bn1=nn.BatchNorm1d(output_size)
        self.bn2=nn.BatchNorm1d(output_size*3+6*6)
      
    
    def forward(self,des_tensor,tweets_tensor,num_prop,category_prop,num_for_h,edge_index,edge_type):
        des_tensor,tweets_tensor,num_prop,category_prop,num_for_h=self.align(des_tensor,tweets_tensor,num_prop,category_prop,num_for_h)
        gnn_out,exp_loss1=self.gnn_moe(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
        gnn_out=gnn_out[:11826]
        text_out,exp_loss2=self.text_moe.forward_roberta(tweets_tensor,des_tensor)
        text_out=text_out[:11826]
        cat_out,exp_loss3=self.cat_moe(num_for_h,category_prop[:11826])

        outputs=[gnn_out,text_out,cat_out]
        exp_loss=exp_loss1+exp_loss2+exp_loss3
        
        out_tensor=torch.stack(outputs,dim=1)
        out_tensor=self.bn1(out_tensor.permute(0,2,1)).permute(0,2,1)
        y=self.dropout(out_tensor)
        y,attention=self.fusion(out_tensor)
        attention=self.fixed_pooling(attention)
        
        y=torch.cat([y.reshape(len(y),self.output_size*3),attention.reshape(len(attention),-1)],dim=1)
        y=self.bn2(y)
        y=self.mlp_classifier(y)
        return y,exp_loss