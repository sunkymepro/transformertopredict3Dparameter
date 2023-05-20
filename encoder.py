import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import random

class PositionalEncoding(nn.Module):   #position embedding
    def __init__(self, d_model, dropout, max_len=1000):

        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model) 
        position = torch.arange(0, max_len).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2) *    
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  
        pe[:, 1::2] = torch.cos(position * div_term)  
        pe = pe.unsqueeze(0)  
        self.register_buffer('pe', pe)  

    def forward(self, x, label=None):
        if label:
            x = x + Variable(self.pe[:, label],requires_grad=False)
        else:
            x = x + Variable(self.pe[:, :x.size(1)],requires_grad=False) #size = [batch, L, d_model]
        return self.dropout(x) # size = [batch, L, d_model]

class MaskGenerator(nn.Module):   
    """Mask generator."""    #用于给tgt embedding加mask，按照一定的比例（比如0， 0.25， 0.5）对tgt的mask序列中随机一部分或者固定的后面部分加mask

    def __init__(self, num_tokens, mask_ratio):
        super().__init__()
        self.num_tokens = num_tokens
        self.mask_ratio = mask_ratio
        self.sort = True

    def uniform_rand(self, length, rate=None):#给定一段长度（tgt序列的长度）输出mask掉的部分的下表和不mask掉的部分的索引
        mask = list(range(int(length)))
        random.shuffle(mask)
        if rate is None:
            rate = self.mask_ratio
        mask_len = int(self.num_tokens * rate)
        self.masked_tokens = mask[self.num_tokens-mask_len:]
        self.unmasked_tokens = mask[:self.num_tokens-mask_len]
        if self.sort:
            self.masked_tokens = sorted(self.masked_tokens)
            self.unmasked_tokens = sorted(self.unmasked_tokens)
        return self.unmasked_tokens, self.masked_tokens

    def forward(self, length, rate=None):
        self.unmasked_tokens, self.masked_tokens = self.uniform_rand(length, rate)
        return self.unmasked_tokens, self.masked_tokens

class nlp_transformer_backbone_mask(nn.Module):
    def __init__(self, length=6, nhead=2, ratio=0.25):
        super(nlp_transformer_backbone_mask, self).__init__()
        self.embedding = nn.Sequential(
                             nn.Linear(6,6),
                             nn.ReLU(),
                             nn.Linear(6,6),
                             nn.ReLU(),
                         )
        self.position = PositionalEncoding(6, 0)
        self.output_layer = nn.Sequential(
                             nn.Linear(6,6),
                             nn.ReLU(),
                             nn.Linear(6,6),
                         )
        self.ratio = ratio
        encoder_layers = nn.TransformerEncoderLayer(6, 3)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, 4)
        decoder_layers = nn.TransformerDecoderLayer(6, 3)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, 1)
        self.mask_gen = MaskGenerator(20, self.ratio)
        torch.nn.init.kaiming_uniform_(self.embedding[0].weight, a=0, mode='fan_out')
        torch.nn.init.kaiming_uniform_(self.embedding[2].weight, a=0, mode='fan_out')
        torch.nn.init.kaiming_uniform_(self.output_layer[0].weight, a=0, mode='fan_out')
        torch.nn.init.kaiming_uniform_(self.output_layer[2].weight, a=0, mode='fan_out')
        for i in list(self.transformer_encoder.parameters()): 
            if i.ndim>=2:
                torch.nn.init.kaiming_uniform_(i, a=0, mode='fan_out')    
        for i in list(self.transformer_decoder.parameters()): 
            if i.ndim>=2:
                torch.nn.init.kaiming_uniform_(i, a=0, mode='fan_out')      

    def forward(self, src, tgt, rate=None, ifmask=True):
        src = self.embedding(src)
        tgt = self.embedding(tgt)#将原始数据处理一下，得到embedding
        unmask_label, mask_label = self.mask_gen(tgt.shape[1], rate)#得到要mask掉的embedding的索引和不mask掉的embedding的索引
        mask_embedding = torch.zeros(tgt.shape)#mask掉的部分用一个值为0的向量代替
        mask_embedding = mask_embedding.to(tgt.device)
        tgt_unmask = tgt[:,unmask_label,:]#将不mask的tgtembedding和src embedding用transformer encoder处理
        tgt_unmask = self.position(tgt_unmask, unmask_label)#加上position embedding
        tgt_unmask = tgt_unmask.permute(1, 0, 2)
        tgt_unmask = self.transformer_encoder(tgt_unmask)
        mask_embedding = mask_embedding[:,:len(mask_label),:]
        mask_embedding = self.position(mask_embedding, mask_label)#给补上mask掉的tgt的0向量加上对应位置的position embedding
        mask_embedding = mask_embedding.permute(1, 0, 2)
        tgt = torch.cat([tgt_unmask, mask_embedding], axis=0)
        src = self.position(src)
        src = src.permute(1, 0, 2)
        src = self.transformer_encoder(src)
        output = self.transformer_decoder(tgt, src)#输入到transformer decoder中
        output = self.output_layer(output)
        return output