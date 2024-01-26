''' Define the Layers '''
import torch.nn as nn
import torch
from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward


__author__ = "Yu-Hsiang Huang"


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout) # 引入多头注意力
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout) # 引入一个前馈神经网络

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask) # 进行encoder中的self-attition操作，qkv均为同一个值
        enc_output = self.pos_ffn(enc_output) # 前馈网络
        return enc_output, enc_slf_attn 


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout) # 带掩码的自注意力
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout) # 交叉注意力
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout) # 前馈网络

    def forward(
            self, dec_input, enc_output,
            slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask) # 引入掩码（长度掩码+自注意力掩码） 此处是  slf_attn_mask= trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq) 
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask) # 由于encoder的长度可能和decoder中的并不一致，因此需要引入掩码
        ## src_pad_idx = opt.trg_pad_idx = field.vocab.stoi[Constants.PAD_WORD] 
        ## dec_enc_attn_mask： src_mask = get_pad_mask(src_seq, src_pad_idx) 使用了encoder也就是输入的掩码
        ## 我的理解是由于encoder是在交叉注意力中扮演kv的角色，因此q会对每一个k求内积，若当前k是代表着掩码，那就使用encoder中的掩码列表对掩码字段进行极小化即可
        ## 因此无需decoder的mask列表，只需encoder中的mask进行处理即可
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn
