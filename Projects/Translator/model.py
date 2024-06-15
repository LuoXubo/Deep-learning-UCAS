import torch.nn.functional as F
import math
import torch
from nltk.tokenize import word_tokenize
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import copy
import numpy as np
import os
import re
import sacrebleu
import random
import time
import jieba
from nltk.tokenize.treebank import TreebankWordDetokenizer

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 下面老是报错 shape 不一致


class Embedding(nn.Module):
    # 词嵌入层
    def __init__(self, d_model, vocab):
        """
        词嵌入层初始化
        :param d_model: 词嵌入维度
        :param vocab: 词表大小
        """
        super(Embedding, self).__init__()
        # Embedding层
        self.lut = nn.Embedding(vocab, d_model)
        # Embedding维数
        self.d_model = d_model

    def forward(self, x):
        # 返回x的词向量（需要乘以math.sqrt(d_model)）
        return self.lut(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    # 位置编码器层
    def __init__(self, d_model, dropout=0.1, max_len=5000, device='cuda'):
        """
        位置编码器层初始化
        :param d_model: 词嵌入维度
        :param dropout: dropout比例
        :param max_len: 序列最大长度
        :param device: 训练设备
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 位置编码矩阵，维度[max_len, embedding_dim]
        pe = torch.zeros(max_len, d_model, device=device)
        # 单词位置
        position = torch.arange(0.0, max_len, device=device)
        position.unsqueeze_(1)
        # 使用exp和log实现幂运算
        div_term = torch.exp(torch.arange(0.0, d_model, 2, device=device) * (- math.log(1e4) / d_model))
        div_term.unsqueeze_(0)
        # 计算单词位置沿词向量维度的纹理值
        pe[:, 0:: 2] = torch.sin(torch.mm(position, div_term))
        pe[:, 1:: 2] = torch.cos(torch.mm(position, div_term))
        # 增加批次维度，[1, max_len, embedding_dim]
        pe.unsqueeze_(0)
        # 将位置编码矩阵注册为buffer(不参加训练)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 将一个批次中语句所有词向量与位置编码相加
        # 注意，位置编码不参与训练，因此设置requires_grad=False
        x += Variable(self.pe[:, : x.size(1), :], requires_grad=False)
        return self.dropout(x)
    
class MultiHeadedAttention(nn.Module):
    # 多头注意力机制
    def __init__(self, h, d_model, dropout=0.1):
        """
        多头注意力机制初始化
        :param h: 多头
        :param d_model: 词嵌入维度
        :param dropout: dropout比例
        """
        super(MultiHeadedAttention, self).__init__()
        # 确保整除
        assert d_model % h == 0
        # q、k、v向量维数
        self.d_k = d_model // h
        # 头的数量
        self.h = h
        # WQ、WK、WV矩阵及多头注意力拼接变换矩阵WO 4个线性层
        self.linears = nn.ModuleList([copy.deepcopy(nn.Linear(d_model, d_model)) for _ in range(4)])
        # 注意力机制函数
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        反向传播
        :param query: q
        :param key: k
        :param value: v
        :param mask: 掩码
        :return:
        """
        if mask is not None:
            mask = mask.unsqueeze(1)
        # 批次大小
        nbatches = query.size(0)
        # WQ、WK、WV分别对词向量线性变换，并将结果拆成h块
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]
        # 注意力加权
        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        # 多头注意力加权拼接
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        # 对多头注意力加权拼接结果线性变换
        return self.linears[-1](x)

    @staticmethod
    def attention(query, key, value, mask=None, dropout=None):
        """
        注意力加权
        :param query: q
        :param key: k
        :param value: v
        :param mask: 掩码矩阵
        :param dropout: dropout比例
        :return:
        """
        # q、k、v向量长度为d_k
        d_k = query.size(-1)
        # 矩阵乘法实现q、k点积注意力，sqrt(d_k)归一化
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        # 注意力掩码机制
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        # 注意力矩阵softmax归一化
        p_attn = F.softmax(scores, dim=-1)
        # dropout
        if dropout is not None:
            p_attn = dropout(p_attn)
        # 注意力对v加权
        return torch.matmul(p_attn, value), p_attn
    
class SublayerConnection(nn.Module):
    # 子层连接结构 用于连接注意力机制以及前馈全连接网络
    def __init__(self, d_model, dropout):
        """
        子层连接结构初始化层
        :param d_model: 词嵌入纬度
        :param dropout: dropout比例
        """
        super(SublayerConnection, self).__init__()
        # 规范化层
        self.norm = nn.LayerNorm(d_model)
        # dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # 层归一化
        x_ = self.norm(x)
        x_ = sublayer(x_)
        x_ = self.dropout(x_)
        # 残差连接
        return x + x_
    
class FeedForward(nn.Module):
    # 前馈全连接网络
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        前馈全连接网络初始化层
        :param d_model: 词嵌入维度
        :param d_ff: 中间隐层维度
        :param dropout: dropout比例
        """
        super(FeedForward, self).__init__()
        # 全连接层
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        # Dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.w_2(x)
        return x
    
class Encoder(nn.Module):
    # 编码器
    def __init__(self, h, d_model, d_ff=256, dropout=0.1):
        """
        编码器层初始化
        :param h: 头数
        :param d_model: 词嵌入维度
        :param d_ff: 中间隐层维度
        :param dropout: dropout比例
        """
        super(Encoder, self).__init__()
        # 多头注意力
        self.self_attn = MultiHeadedAttention(h, d_model)
        # 前馈全连接层
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        # 子层连接结构
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)
        # 规范化层
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        # 将embedding层进行Multi head Attention
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, mask))
        # attn的结果直接作为下一层输入
        return self.norm(self.sublayer2(x, self.feed_forward))
    
class Decoder(nn.Module):
    def __init__(self, h, d_model, d_ff=256, dropout=0.1):
        """
        解码器层
        :param h: 头数
        :param d_model: 词嵌入维度
        :param d_ff: 中间隐层维度
        :param dropout: dropout比例
        """
        super(Decoder, self).__init__()
        self.size = d_model
        # 自注意力机制
        self.self_attn = MultiHeadedAttention(h, d_model)
        # 上下文注意力机制
        self.src_attn = MultiHeadedAttention(h, d_model)
        # 前馈全连接子层
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        # 子层连接结构
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)
        self.sublayer3 = SublayerConnection(d_model, dropout)
        # 规范化层
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, memory, src_mask, tgt_mask):
        # memory为编码器输出隐表示
        m = memory
        # 自注意力机制，q、k、v均来自解码器隐表示
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 上下文注意力机制：q为来自解码器隐表示，而k、v为编码器隐表示
        x = self.sublayer2(x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.norm(self.sublayer3(x, self.feed_forward))
    
class Generator(nn.Module):
    #  生成器层
    def __init__(self, d_model, vocab):
        """
        生成器层初始化
        :param d_model:
        :param vocab:
        """
        super(Generator, self).__init__()
        # decode后的结果，先进入一个全连接层变为词典大小的向量
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        # 然后再进行log_softmax操作(在softmax结果上再做多一次log运算)
        return F.log_softmax(self.proj(x), dim=-1)
    
class Transformer(nn.Module):
    # Transformer层
    def __init__(self, tokenizer, h=8, d_model=256, E_N=2, D_N=2, device='cuda'):
        """
        transformer层初始化
        :param h: 头数
        :param d_model: 词嵌入纬度
        :param tokenizer:
        :param E_N:
        :param D_N:
        :param device:
        """
        super(Transformer, self).__init__()
        # 编码器
        self.encoder = nn.ModuleList([Encoder(h, d_model) for _ in range(E_N)])
        # 解码器
        self.decoder = nn.ModuleList([Decoder(h, d_model) for _ in range(D_N)])
        # 词嵌入层
        self.src_embed = Embedding(d_model, tokenizer.get_vocab_size())
        self.tgt_embed = Embedding(d_model, tokenizer.get_vocab_size())
        # 位置编码器层
        self.src_pos = PositionalEncoding(d_model, device=device)
        self.tgt_pos = PositionalEncoding(d_model, device=device)
        # 生成器层
        self.generator = Generator(d_model, tokenizer.get_vocab_size())

    def encode(self, src, src_mask):
        """
        编码
        :param src: 源数据
        :param src_mask: 源数据掩码
        :return:
        """
        
        # 词嵌入
        src = self.src_embed(src)
        # 位置编码
        src = self.src_pos(src)
        # 编码
        for i in self.encoder:
            src = i(src, src_mask)
        return src

    def decode(self, memory, tgt, src_mask, tgt_mask):
        """
        解码
        :param memory: 编码器输出
        :param tgt: 目标数据输入
        :param src_mask: 源数据掩码
        :param tgt_mask: 目标数据掩码
        :return:
        """
        #  词嵌入
        tgt = self.tgt_embed(tgt)
        #  位置编码
        tgt = self.tgt_pos(tgt)
        # 解码
        for i in self.decoder:
            tgt = i(tgt, memory, src_mask, tgt_mask)
        return tgt

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        反向传播
        :param src: 源数据
        :param tgt: 目标数据
        :param src_mask: 源数据掩码
        :param tgt_mask: 目标数据掩码
        :return:
        """
        # encoder的结果作为decoder的memory参数传入，进行decode
        return self.decode(self.encode(src, src_mask), tgt, src_mask, tgt_mask)