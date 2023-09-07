import torch
from torch import nn
from torch_geometric.nn.conv import GCNConv
from torch_geometric.loader import NeighborSampler



class LModel(nn.Module):
    def __init__(self, embed_dim=768, num_heads=2, dropout=0.3, activation='ReLU',
                 norm_first=True, layer_norm_eps=1e-5,exp=2,k=1):
        super(LModel, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,
                                                         dropout=dropout, batch_first=True)
        if activation == 'ReLU':
            self.activation = nn.ReLU()
        if activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU()
        if activation == 'SELU':
            self.activation = nn.SELU()
        '''
        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(embed_dim, embed_dim)
        '''
        #self.moe=MoE(embed_dim*3,embed_dim*3,exp,embed_dim*3,model=MLP,k=k)
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, text_src):
        if self.norm_first:
            text, attention_weight = self._sa_block(self.norm1(text_src))
            #text = text_src + text
            #text1,loss=self._ff_block(self.norm2(text))
            #text = text + text1
        else:
            text, attention_weight = self._sa_block(text_src)
            text = self.norm1(text_src + text)
            text1,loss=self._ff_block(self.norm2(text))
            text = self.norm2(text + text1)
        return text, attention_weight

    def _sa_block(self, text):
        text, attention_weight = self.multihead_attention(text, text, text)
        text = self.dropout1(text)
        return text, attention_weight

    def _ff_block(self, text):
        #text = self.linear2(self.dropout(self.activation(self.linear1(text))))
        text_len=text.shape[1]
        text,loss=self.moe(text.reshape((len(text),-1)))
        text = self.dropout2(text)
        return text.reshape((len(text),text_len,-1)),loss
