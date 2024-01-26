import torch
import torch.nn as nn
import torch.nn.functional as F

__author__ = "Yu-Hsiang Huang"

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    # 实现注意力机制的点积算法

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature #sqrt（dk） 即为对qk的乘积进行scale操作
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3)) #transpose函数将k的2,3维进行了交换  生成权重向量
        # b x n x lq x dv @ b x n x dv x lq = b x n x lq x lq

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9) #若是decoder中的带掩码的自注意力机制或者是存在无意义的占位符，则需要进行mask操作

        attn = self.dropout(F.softmax(attn, dim=-1)) #Dropout 操作会随机将一些元素置零，以减小网络对某些特定输入的依赖性，从而提高模型的泛化能力
        output = torch.matmul(attn, v) #类似于对v向量进行加权和  b x n x lq x dv

        return output, attn #返回输出向量和权重向量
