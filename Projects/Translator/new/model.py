"""
@Description :   
@Author      :   Xubo Luo 
@Time        :   2024/06/15 14:57:45
"""

import torch.nn as nn
import torch
import copy
import torch.nn.functional as F
from torch.autograd import Variable
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clones(module, N):
    """
    克隆模型块，克隆的模型块参数不共享
    """
    return nn.ModuleList([copy.deepcopy(module).to(DEVICE) for _ in range(N)])

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        # 保证可以整除
        assert d_model % h == 0
        # 得到一个head的attention表示维度
        self.d_k = d_model // h
        # head数量
        self.h = h
        # 定义4个全连接函数，供后续作为WQ，WK，WV矩阵和最后h个多头注意力矩阵concat之后进行变换的矩阵
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.to(DEVICE)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        # query的第一个维度值为batch size
        nbatches = query.size(0)
        # print('nbatches=', nbatches)
        # 将embedding层乘以WQ，WK，WV矩阵(均为全连接)
        # 并将结果拆成h块，然后将第二个和第三个维度值互换(具体过程见上述解析)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
        # 调用上述定义的attention函数计算得到h个注意力矩阵跟value的乘积，以及注意力矩阵
        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        # 将h个多头注意力矩阵concat起来（注意要先把h变回到第三维的位置）
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        # 使用self.linears中构造的最后一个全连接函数来存放变换后的矩阵进行返回
        return self.linears[-1](x)

    def attention(self, query, key, value, mask=None, dropout=None):
        # 将query矩阵的最后一个维度值作为d_k
        d_k = query.size(-1)

        # TODO: 将key的最后两个维度互换(转置)，才能与query矩阵相乘，乘完了还要除以d_k开根号
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # 如果存在要进行mask的内容，则将那些为0的部分替换成一个很大的负数
        scores = scores.to(DEVICE)
        mask = mask.to(DEVICE)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # TODO: 将mask后的attention矩阵按照最后一个维度进行softmax
        p_attn = F.softmax(scores, dim=-1)

        # 如果dropout参数设置为非空，则进行dropout操作
        if dropout is not None:
            p_attn = dropout(p_attn)
        # 最后返回注意力矩阵跟value的乘积，以及注意力矩阵
        return torch.matmul(p_attn, value), p_attn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.to(DEVICE)

    def forward(self, x):
        # TODO: 请利用init中的成员变量实现Feed Forward层的功能
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # 初始化α为全1, 而β为全0
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        # 平滑项
        self.eps = eps
        self.to(DEVICE)

    def forward(self, x):
        # TODO: 请利用init中的成员变量实现LayerNorm层的功能
        # 按最后一个维度计算均值和方差
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        # TODO: 返回Layer Norm的结果
        return self.a_2 * (x - mean) / torch.sqrt(std ** 2 + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    SublayerConnection的作用就是把Multi-Head Attention和Feed Forward层连在一起
    只不过每一层输出之后都要先做Layer Norm再残差连接
    sublayer是lamda函数
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.to(DEVICE)

    def forward(self, x, sublayer):
        # TODO: 请利用init中的成员变量实现LayerNorm和残差连接的功能
        # 返回Layer Norm和残差连接后结果
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    def __init__(self, h, d_model, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(h, d_model).to(DEVICE)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout).to(DEVICE)
        # SublayerConnection的作用就是把multi和ffn连在一起
        # 只不过每一层输出之后都要先做Layer Norm再残差连接
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        # d_model
        self.size = d_model
        self.to(DEVICE)

    def forward(self, x, mask):
        # 将embedding层进行Multi head Attention
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # 注意到attn得到的结果x直接作为了下一层的输入
        return self.sublayer[1](x, self.feed_forward)

class Encoder(nn.Module):
    # layer = EncoderLayer
    # N = 6
    def __init__(self, h, d_model, d_ff, dropout, N):
        super(Encoder, self).__init__()
        # 复制N个encoder layer
        layer = EncoderLayer(h, d_model, d_ff, dropout).to(DEVICE)
        self.layers = clones(layer, N)
        # Layer Norm
        self.norm = LayerNorm(d_model).to(DEVICE)
        self.to(DEVICE)

    def forward(self, x, mask):
        """
        使用循环连续eecode N次(这里为6次)
        这里的Eecoderlayer会接收一个对于输入的attention mask处理
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self, h, d_model, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.size = d_model
        # Self-Attention
        self.self_attn = MultiHeadedAttention(h, d_model).to(DEVICE)
        # 与Encoder传入的Context进行Attention
        self.src_attn = MultiHeadedAttention(h, d_model).to(DEVICE)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer = clones(SublayerConnection(d_model, dropout), 3)
        self.to(DEVICE)

    def forward(self, x, memory, src_mask, tgt_mask):
        # 用m来存放encoder的最终hidden表示结果
        m = memory

        # TODO: 参照EncoderLayer完成DecoderLayer的forwark函数
        # Self-Attention：注意self-attention的q，k和v均为decoder hidden
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # Context-Attention：注意context-attention的q为decoder hidden，而k和v为encoder hidden
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

class Decoder(nn.Module):
    def __init__(self, h, d_model, d_ff, dropout, N, vocab):
        super(Decoder, self).__init__()
        layer = DecoderLayer(h, d_model, d_ff, dropout)
        # 复制N个encoder layer
        self.layers = clones(layer, N)
        # Layer Norm
        self.norm = LayerNorm(layer.size)
        self.proj = nn.Linear(d_model, vocab)
        self.to(DEVICE)

    def forward(self, x, memory, src_mask, tgt_mask, voc_out=False):
        """
        使用循环连续decode N次(这里为6次)
        这里的Decoderlayer会接收一个对于输入的attention mask处理
        和一个对输出的attention mask + subsequent mask处理
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        x = self.norm(x)
        if not voc_out:
            return x
        else:
            return F.log_softmax(self.proj(x), dim=-1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, DEVICE, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化一个size为 max_len(设定的最大长度)×embedding维度 的全零矩阵
        # 来存放所有小于这个长度位置对应的porisional embedding
        pe = torch.zeros(max_len, d_model, device=DEVICE)
        # 生成一个位置下标的tensor矩阵(每一行都是一个位置下标)
        """
        形式如：
        tensor([[0.],
                [1.],
                [2.],
                [3.],
                [4.],
                ...])
        """
        position = torch.arange(0., max_len, device=DEVICE).unsqueeze(1)
        # 这里幂运算太多，我们使用exp和log来转换实现公式中pos下面要除以的分母（由于是分母，要注意带负号）
        div_term = torch.exp(torch.arange(0., d_model, 2, device=DEVICE) * -(math.log(10000.0) / d_model))

        # TODO: 根据公式，计算各个位置在各embedding维度上的位置纹理值，存放到pe矩阵中
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 加1个维度，使得pe维度变为：1×max_len×embedding维度
        # (方便后续与一个batch的句子所有词的embedding批量相加)
        pe = pe.unsqueeze(0)
        # 将pe矩阵以持久的buffer状态存下(不会作为要训练的参数)
        self.register_buffer('pe', pe)
        self.to(DEVICE)

    def forward(self, x):
        # 将一个batch的句子所有词的embedding与已构建好的positional embeding相加
        # (这里按照该批次数据的最大句子长度来取对应需要的那些positional embedding值)
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab, DEVICE):
        super(Embeddings, self).__init__()
        # Embedding层
        self.lut = nn.Embedding(vocab, d_model).to(DEVICE)
        # Embedding维数
        self.d_model = d_model
        self.DEVICE = DEVICE
        self.to(DEVICE)

    def forward(self, x):
        # 返回x对应的embedding矩阵（需要乘以math.sqrt(d_model)）
        x = x.to(self.DEVICE)
        return self.lut(x) * math.sqrt(self.d_model)

# class Generator(nn.Module):
#     # vocab: tgt_vocab
#     def __init__(self, d_model, vocab):
#         super(Generator, self).__init__()
#         # decode后的结果，先进入一个全连接层变为词典大小的向量
#         self.proj = nn.Linear(d_model, vocab)
#
#     def forward(self, x):
#         # 然后再进行log_softmax操作(在softmax结果上再做多一次log运算)
#         return F.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    def __init__(self, h, d_model, d_ff, dropout, N, src_vocab, tgt_vocab, DEVICE):
        super(Transformer, self).__init__()
        self.encoder = Encoder(h, d_model, d_ff, dropout, N).to(DEVICE)
        self.decoder = Decoder(h, d_model, d_ff, dropout, N, tgt_vocab).to(DEVICE)
        self.src_embed = nn.Sequential(Embeddings(d_model, src_vocab, DEVICE).to(DEVICE), PositionalEncoding(d_model, dropout, DEVICE).to(DEVICE))
        self.tgt_embed = nn.Sequential(Embeddings(d_model, tgt_vocab, DEVICE).to(DEVICE), PositionalEncoding(d_model, dropout, DEVICE).to(DEVICE))
        # self.generator = Generator(d_model, tgt_vocab)

        for p in self.parameters():
            if p.dim() > 1:
                # 这里初始化采用的是nn.init.xavier_uniform
                nn.init.xavier_uniform_(p)
        self.to(DEVICE)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask, voc_out=True)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # encoder的结果作为decoder的memory参数传入，进行decode
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
