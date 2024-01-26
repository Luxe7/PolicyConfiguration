''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformer.Modules import ScaledDotProductAttention

__author__ = "Yu-Hsiang Huang"

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):

        super().__init__()

        self.n_head = n_head #头数
        self.d_k = d_k #key和query的向量维度
        self.d_v = d_v #value的向量维度

        # 三个线性层做矩阵乘法生成q, k, v
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False) 
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False) # 将多个头学习到的向量通过一个线性层，实现多头学习到内容的拼接

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5) #注意力计算

        self.dropout = nn.Dropout(dropout) # 丢弃率
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6) # 这里使用layernorm的正则化方式，对比batch不容易受到时序数据长度的影响


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head 
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1) 
        # sz_b指的是batchsize，len_q, len_k, len_v指的是三者的第二维大小，也就是每一个样本的长度（seq序列长度

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        #b: batch_size, lq: translation task的seq长度, n: head数, dv: embedding vector length
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            # (batchSize, 1, seqLen) -> (batchSize, 1, 1, seqLen)
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask) # b x n x lq x dv

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        # 将多个头学习到的连在一起
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1) # .contiguous()：这是为了保证内存中的连续性 
        
        # 将连接的结果再进入一层线性层进行投影 其实这这个线性层的输入和输出维度在论文中是一致的，
        #但是其实在实际操作中可以不一致，该层也可以将最终的维度确定在d_model的大小，论文中是512
        q = self.dropout(self.fc(q)) 

        q += residual #残差连接
        q = self.layer_norm(q) #进行norm操作

        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''
    # 这个地方是transformer中引入非线性的部分，前面的attention全是线性操作

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6) # 进行layernorm
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual #add

        x = self.layer_norm(x) #norm

        return x
