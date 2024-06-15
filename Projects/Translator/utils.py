"""
@Description :   
@Author      :   Xubo Luo 
@Time        :   2024/06/15 14:12:19
"""

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

class TranslationDataset(Dataset):
    # 创建数据集
    def __init__(self, src, tgt):
        """
        初始化
        :param src: 源数据(经tokenizer处理后)
        :param tgt: 目标数据(经tokenizer处理后)
        """
        self.src = src
        self.tgt = tgt

    def __getitem__(self, i):
        return self.src[i], self.tgt[i]

    def __len__(self):
        return len(self.src)
    

class Tokenizer():
    ## 定义tokenizer,对原始数据进行处理
    def __init__(self, en_path, ch_path, count_min=5):
        """
        初始化
        :param en_path: 英文数据路径
        :param ch_path: 中文数据路径
        :param count_min: 对出现次数少于这个次数的数据进行过滤
        """
        self.en_path = en_path  # 英文路径
        self.ch_path = ch_path  # 中文路径
        self.__count_min = count_min  # 对出现次数少于这个次数的数据进行过滤

        # 读取原始英文数据
        self.en_data = self.__read_ori_data(en_path)
        # 读取原始中文数据
        self.ch_data = self.__read_ori_data(ch_path)

        self.index_2_word = ['unK', '<pad>', '<bos>', '<eos>']
        self.word_2_index = {'unK': 0, '<pad>': 1, '<bos>': 2, '<eos>': 3}

        self.en_set = set()
        self.en_count = {}

        # 中英文字符计数
        self.__en_count = {}
        self.__ch_count = {}

        self.__count_word()
        self.mx_length = 40
        # 创建英文词汇表
        self.data_ = []
        self.__filter_data()
        random.shuffle(self.data_)
        self.test = self.data_[-1000:]
        self.data_ = self.data_[:-1000]

    def __read_ori_data(self, path):
        """
        读取原始数据
        :param path: 数据路径
        :return: 返回一个列表，每个元素是一条数据
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = f.read().split('\n')[:-1]
        return data

    def __count_word(self):
        """
        统计中英文词汇表
        :return:
        """
        le = len(self.en_data)
        p = 0
        # 统计英文词汇表
        for data in self.en_data:
            if p % 1000 == 0:
                print('英文', p / le)
            sentence = word_tokenize(data)
            for sen in sentence:
#                 sen=sen.lower()
                if sen in self.en_set:
                    self.en_count[sen] += 1
                else:
                    self.en_set.add(sen)
                    self.en_count[sen] = 1
            p += 1
        for k, v in self.en_count.items():
            if v >= self.__count_min:
                self.word_2_index[k] = len(self.index_2_word)
                self.index_2_word.append(k)
            else:
                self.word_2_index[k] = 0
        self.en_set = set()
        self.en_count = {}
        p = 0
        # 统计中文词汇表
        for data in self.ch_data:
            if p % 1000 == 0:
                print('中文', p / le)
            sentence = list(jieba.cut(data))
            for sen in sentence:
                if sen in self.en_set:
                    self.en_count[sen] += 1
                else:
                    self.en_set.add(sen)
                    self.en_count[sen] = 1
            p += 1
        # 构建词汇表
        for k, v in self.en_count.items():
            if v >= self.__count_min:
                self.word_2_index[k] = len(self.index_2_word)
                self.index_2_word.append(k)
            else:
                self.word_2_index[k] = 0

    def __filter_data(self):
        length = len(self.en_data)
        for i in range(length):
            # 0 代表英文到中文，1 代表中文到英文
            self.data_.append([self.en_data[i], self.ch_data[i], 0])
            self.data_.append([self.ch_data[i], self.en_data[i], 1])

    def en_cut(self, data):
        data = word_tokenize(data)
        # 用于存放每个句子对应的编码
        if len(data) > self.mx_length:
            return 0, []
        en_tokens = []
        # 对分词结果进行遍历
        for tk in data:
#             x = tk.lower()
            # 对于结果进行编码,0代表unK
            en_tokens.append(self.word_2_index.get(tk, 0))
        return 1, en_tokens

    def ch_cut(self, data):
        data = list(jieba.cut(data))
#         list(data)[:-1]
        # 用于存放每个句子对应的编码
        if len(data) > self.mx_length:
            return 0, []
        en_tokens = []
        # 对分词结果进行遍历
        for tk in data:
            # 对于结果进行编码,0代表unK
            en_tokens.append(self.word_2_index.get(tk, 0))
        return 1, en_tokens

    def encode_all(self, data):
        """
        对一组数据进行编码
        :param data: data是一个数组，形状为n*3 每个元素是[src_sentence, tgt_sentence, label]，label 0 代表英文到中文，1 代表中文到英文
        :return:
        """
        src = []
        tgt = []
        en_src, en_tgt, l = [], [], []
        labels=[]
        for i in data:
            en_src.append(i[0])
            en_tgt.append(i[1])
            l.append(i[2])
        for i in range(len(l)):
            if l[i] == 0:
                lab1, src_tokens = self.en_cut(en_src[i])
                if lab1 == 0:
                    continue
                lab2, tgt_tokens = self.ch_cut(en_tgt[i])
                if lab2 == 0:
                    continue
                src.append(src_tokens)
                tgt.append(tgt_tokens)
                labels.append(i)
            else:
                lab1, tgt_tokens = self.en_cut(en_tgt[i])
                if lab1 == 0:
                    continue
                lab2, src_tokens = self.ch_cut(en_src[i])
                if lab2 == 0:
                    continue
                src.append(src_tokens)
                tgt.append(tgt_tokens)
                labels.append(i)
        return labels,src, tgt

    def encode(self, src, l):
        if l == 0:
            src1 = word_tokenize(src)
            # 用于存放每个句子对应的编码
            en_tokens = []
            # 对分词结果进行遍历
            for tk in src1:
#                 x = tk.lower()
                # 对于结果进行编码
                en_tokens.append(self.word_2_index.get(tk, 0))
            return [en_tokens]
        else:
            src1 = list(jieba.cut(src))
            # 用于存放每个句子对应的编码
            en_tokens = []
            # 对分词结果进行遍历
            for tk in src1:
                # 对于结果进行编码
                en_tokens.append(self.word_2_index.get(tk, 0))
            return [en_tokens]

    def decode(self, data):
        """
        数据解码
        :param data: 这里传入一个中文的index
        :return: 返回解码后的一个字符
        """
        return self.index_2_word[data]

    def __get_datasets(self, data):
        """
        获取数据集
        :return:返回DataSet类型的数据 或者 None
        """
        # 将数据编码并
        labels,src, tgt = self.encode_all(data)
        # 返回数据集
        return TranslationDataset(src, tgt)

    def another_process(self, batch_datas):
        """
        特殊处理，这里传入一个batch的数据，并对这个batch的数据进行填充，使得每一行的数据长度相同。这里填充pad 空字符  bos 开始  eos结束
        :param batch_datas: 一个batch的数据
        :return: 返回填充后的数据
        """
        # 创建四个空字典存储数据
        en_index, ch_index = [], []  # 中文英文索引，中文索引
        en_len, ch_len = [], []  # 没行英文长度，每行中文长度

        for en, ch in batch_datas:  # 对batch进行遍历，将所有数据的索引与长度加入四个列表
            en_index.append(en)
            ch_index.append(ch)
            en_len.append(len(en))
            ch_len.append(len(ch))

        # 获取中英文的最大长度，根据这个长度对所有数据进行填充，使每行数据长度相同
        max_en_len = max(en_len)
        max_ch_len = max(ch_len)
        max_len = max(max_en_len, max_ch_len + 2)

        # 英文数据填充，i是原始数据，后面是填充的pad
        en_index = [i + [self.word_2_index['<pad>']] * (max_len - len(i)) for i in en_index]
        # 中文数据填充 先填充bos表示句子开始，后面接原始数据，最后填充eos表示句子结束，后面接pad
        ch_index = [[self.word_2_index['<bos>']] + i + [self.word_2_index['<eos>']] +
                    [self.word_2_index['<pad>']] * (max_len - len(i) + 1) for i in ch_index]

        # 将处理后的数据转换为tensor并放到相应设备上
        en_index = torch.tensor(en_index)
        ch_index = torch.tensor(ch_index)
        return en_index, ch_index

    def get_dataloader(self, data, batch_size=40):
        """
        获取dataloader
        :return:
        """
        # 获取数据集
        data = self.__get_datasets(data)
        # 返回DataLoader类型的数据
        return DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=self.another_process)

    # 获取英文词表大小
    def get_vocab_size(self):
        return len(self.index_2_word)

    # 获取数据集大小
    def get_dataset_size(self):
        return len(self.en_data)
    

def subsequent_mask(size):
    """
    注意力机制掩码生成
    :param size: 句子长度
    :return: 注意力掩码
    """
    # 设定subsequent_mask矩阵的shape
    attn_shape = (1, size, size)
    # 生成一个右上角(不含主对角线)为全1，左下角(含主对角线)为全0的subsequent_mask矩阵
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    # 返回一个右上角(不含主对角线)为全False，左下角(含主对角线)为全True的subsequent_mask矩阵
    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    # 批次类,对每一个批次的数据进行掩码生成操作
    def __init__(self, src, trg=None, tokenizer=None, device='cuda'):
        """
        初始化函数
        :param src: 源数据
        :param trg: 目标数据
        :param tokenizer: 分词器
        :param device: 训练设备
        """
        # 将输入、输出单词id表示的数据规范成整数类型并转换到训练设备上
        src = src.to(device).long()
        trg = trg.to(device).long()
        self.src = src  # 源数据 (batch, seq_len)
        self.__pad = tokenizer.word_2_index['<pad>']  # 填充字符的索引
        # 对于当前输入的语句非空部分进行判断，这里是对源数据进行掩码操作，将填充的内容置为0
        # 并在seq length前面增加一维，形成维度为 1×seq length 的矩阵
        self.src_mask = (src != self.__pad).unsqueeze(-2)
        # 如果输出目标不为空，则需要对解码器使用的目标语句进行掩码
        if trg is not None:
            # 解码器使用的目标输入部分
            self.trg = trg[:, : -1]
            # 解码器训练时应预测输出的目标结果
            self.trg_y = trg[:, 1:]
            # 将目标输入部分进行注意力掩码
            self.trg_mask = self.make_std_mask(self.trg, self.__pad)
            # 将应输出的目标结果中实际的词数进行统计
            self.ntokens = (self.trg_y != self.__pad).data.sum()

    # 掩码操作
    @staticmethod
    def make_std_mask(tgt, pad):
        """
        生成掩码矩阵
        :param tgt: 目标数据
        :param pad: 填充字符的索引
        :return:
        """
        tgt_mask = (tgt != pad).unsqueeze(-2)  # 首先对pad进行掩码生成
        # 这里对注意力进行掩码操作并与pad掩码结合起来。
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask
    
class LabelSmoothing(nn.Module):
    # 标签平滑
    def __init__(self, size, padding_idx, smoothing=0.0):
        """
        初始化
        :param size: 目标数据词表大小
        :param padding_idx: 目标数据填充字符的索引
        :param smoothing: 做平滑的值，为0即不进行平滑
        """
        super(LabelSmoothing, self).__init__()
        # KL散度，通常用于测量两个概率分布之间的差异
        self.criterion = nn.KLDivLoss(reduction='sum')
        # 目标数据填充字符的索引
        self.padding_idx = padding_idx
        # 置信度
        self.confidence = 1.0 - smoothing
        # 平滑值
        self.smoothing = smoothing
        # 词表大小
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        """
        反向传播
        :param x: 预测值
        :param target: 目标值
        :return:
        """
        # 判断输出值的第二维传长度是否等于输出词表的大小，这里x的shape为 （batch*seqlength,x.shape(-1)）
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        # 标签平滑填充
        true_dist.fill_(self.smoothing / (self.size - 2))
        # 这里的操作是将真实值的位置进行替换,替换成置信度
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # 将填充的位置的值设置为0
        true_dist[:, self.padding_idx] = 0
        # 生成填充部分的掩码
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        # 返回KL散度
        return self.criterion(x, Variable(true_dist, requires_grad=False))
    
class SimpleLossCompute:
    # 计算损失和进行参数反向传播
    def __init__(self, generator, criterion, opt=None):
        """
        初始化
        :param generator: 生成器，transformer模块中的最后一层，这里将其单独拿出来而不直接放进transformer中的原因是：
            预测数据的是时候，我们需要利用之前的结果，但是我们只去最后一个作为本次输出，那么在进行输出时，只对最后一个进行输出，单独拿出来进行输出的线性变换，更灵活
        :param criterion: 标签平滑的类
        :param opt: 经wormup后的optimizer
        """
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        """
        类做函数调用
        :param x: 经transformer解码后的结果
        :param y: 目标值
        :param norm: 本次数据有效的字符数，即，除去padding后的字符数
        :return:
        """
        # 进行输出
        x = self.generator(x)
        # 得到KL散度
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        # 反向椽笔
        loss.backward()
        if self.opt is not None:
            # 参数更新
            self.opt.step()
            # 优化器梯度置0
            self.opt.optimizer.zero_grad()
        # 返回损失
        return loss.data.item() * norm.float()
    
class NoamOpt:
    # warmup
    def __init__(self, model_size, factor, warmup, optimizer):
        """
        初始化
        :param model_size: 词嵌入维度
        :param factor:
        :param warmup:
        :param optimizer:
        """
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        # 学习率更新
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        self.optimizer.zero_grad()

    def rate(self, step=None):
        # 学习率更新函数
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

def is_english_sentence(sentence):
    # 使用正则表达式检查句子中是否包含英文字母
    english_pattern = re.compile(r'[a-zA-Z]')
    match = english_pattern.search(sentence)
    # True 表示这是英文句子
    if match: 
        return True
    else:
        return False
    

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

# 这个smooth防止句子长度小于4而出现报错
smooth = SmoothingFunction().method1
def compute_bleu4(tokenizer, random_integers, model, device):
    """
    计算BLEU4
    :param tokenizer: tokenizer
    :param random_integers: 这个是随机选择的测试集数据的编号
    :param model: 模型
    :param device: 设备
    :return:
    """
    # m1,m2存放英文的原数据与模型输出数据
    m1, m2 = [], []
    # m3,m4存放英文的原数据与模型输出数据
    m3, m4 = [], []
    model.eval()
    # 存放测试数据
    da = []
    # 将随机选择的测试集数据编号添加到da中
    for i in random_integers:
        da.append(tokenizer.test[i])
    # 对da中的数据进行编码
    labels, x, _ = tokenizer.encode_all(da)
    with torch.no_grad():
        # 预测
        y = predict(x, model, tokenizer, device)
    # 这个p用于记录y的索引
    p = 0
    # 用于保存有效的索引
    itg = []
    # 这里我限制输入数据全部有效，如果有无效的数据，直接放弃本次计算
    if len(y) != 10:
        return 0
    for i in labels:
        # 取出有效的索引
        itg.append(random_integers[i])
    # 将真实数据与预测数据分别放到m1,m2,m3,m4中
    for i in itg:
        if is_english_sentence(tokenizer.test[i][1]):
            m1.append(tokenizer.test[i][1])
            m2.append([y[p]])
        else:
            m3.append(list(jieba.cut(tokenizer.test[i][1])))
            m4.append([list(jieba.cut(y[p]))])
        p += 1
    smooth = SmoothingFunction().method1
    # 计算英文的bleu4
    b1 = [sacrebleu.sentence_bleu(candidate, refs).score for candidate, refs in zip(m1, m2)]
    # 计算中文的bleu4
    for i in range(len(m4)):
        b2 = sentence_bleu(m4[i], m3[i], weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth) * 100
        b1.append(b2)
#     print(b1)
#     print(sum(b1)/len(b1))
    return sum(b1)/len(b1)


from nltk.corpus import words
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """
    传入一个训练好的模型，对指定数据进行预测
    """
    # 先用encoder进行encode
    memory = model.encode(src, src_mask)
    # 初始化预测内容为1×1的tensor，填入开始符('BOS')的id，并将type设置为输入数据类型(LongTensor)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    # 遍历输出的长度下标
    for i in range(max_len - 1):
        # decode得到隐层表示
        out = model.decode(memory,
                           Variable(ys),
                           src_mask,
                           Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        # 将隐藏表示转为对词典各词的log_softmax概率分布表示
        prob = model.generator(out[:, i])
        # 获取当前位置最大概率的预测词id
        _, next_word = torch.max(prob, dim=-1)
        next_word = next_word.data[0]
        # 将当前位置预测的字符id与之前的预测内容拼接起来
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

def predict(data, model, tokenizer, device='cuda'):
    """
    在data上用训练好的模型进行预测，打印模型翻译结果
    """
    # 梯度清零
    with torch.no_grad():
        # 在data的英文数据长度上遍历下标
        data1=[]
        for i in range(len(data)):
            # 打印待翻译的英文语句

            # 将当前以单词id表示的英文语句数据转为tensor，并放如DEVICE中
            src = torch.from_numpy(np.array(data[i])).long().to(device)
            # 增加一维
            src = src.unsqueeze(0)
            # 设置attention mask
            src_mask = (src != tokenizer.word_2_index['<pad>']).unsqueeze(-2)
            # 用训练好的模型进行decode预测
            out = greedy_decode(model, src, src_mask, max_len=100, start_symbol=tokenizer.word_2_index['<bos>'])
            # 初始化一个用于存放模型翻译结果语句单词的列表
            translation = []
            # 遍历翻译输出字符的下标（注意：开始符"BOS"的索引0不遍历）
            for j in range(1, out.size(1)):
                # 获取当前下标的输出字符
                sym = tokenizer.index_2_word[out[0, j].item()]
                # 如果输出字符不为'EOS'终止符，则添加到当前语句的翻译结果列表
                if sym != '<eos>':
                    translation.append(sym)
                # 否则终止遍历
                else:
                    break
            # 打印模型翻译输出的中文语句结果
            if len(translation)>0:
                if translation[0].lower() in words.words():
                    data1.append(TreebankWordDetokenizer().detokenize(translation))
                else:
                    data1.append("".join(translation))
        return data1