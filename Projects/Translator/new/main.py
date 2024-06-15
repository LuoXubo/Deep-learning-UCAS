"""
@Description :   
@Author      :   Xubo Luo 
@Time        :   2024/06/15 14:57:53
"""

import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import os

import random
# from data_generator import *
from utils import *
import numpy as np
# from transformer_models import *
from model import *
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomLearnRateOptimizer:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, model):
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

class LabelSmoothingLoss(nn.Module):
    """标签平滑处理"""

    def __init__(self, size, padding_idx=0, smoothing=0.0):
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        x = x.to(DEVICE)
        target = target.to(DEVICE)
        true_dist = x.data.clone()

        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

class Trainer:
    def __init__(self, epochs, tgt_vocab, D_MODEL, model, SAVE_FILE='model1.pt', MAX_LENGTH=100):
        self.epochs = epochs
        self.loss = LabelSmoothingLoss(tgt_vocab)
        self.optimizer = CustomLearnRateOptimizer(D_MODEL, 1, 2000, model)
        self.SAVE_FILE = SAVE_FILE
        self.MAX_LENGTH = MAX_LENGTH
        self.model = model

    def compute_loss(self, x, y, norm):
        # 计算loss，并且反向传播
        loss = self.loss(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm
        # print('loss=', loss.shape, loss)
        loss.backward()
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.optimizer.zero_grad()
        return loss.data.item() * norm.float()

    def train(self, data):
        """
        训练并保存模型
        """
        for p in self.model.parameters():
            if p.dim() > 1:
                # 这里初始化采用的是nn.init.xavier_uniform
                nn.init.xavier_uniform_(p)
        # 初始化模型在dev集上的最优Loss为一个较大值
        best_dev_loss = 1e5

        for epoch in range(self.epochs):
            # 模型训练
            self.model.train()
            self.run_epoch(data.train_data, self.model, epoch)
            self.model.eval()

            # 在dev集上进行loss评估
            print('>>>>> Evaluate')
            dev_loss = self.run_epoch(data.dev_data, self.model, epoch, max_step=200)
            print('<<<<< Evaluate loss: %f' % dev_loss)
            # # TODO: 如果当前epoch的模型在dev集上的loss优于之前记录的最优loss则保存当前模型，并更新最优loss值
            if dev_loss < best_dev_loss:
                print('saving model...')
                torch.save(self.model.state_dict(), self.SAVE_FILE)
                best_dev_loss = dev_loss
            print('>>>>>>>>>>>>>>> Evaluate case, epach=%d' % epoch)
            self.evaluate(data)
            print('>>>>>>>>>>>>>>> end evaluate case, epach=%d' % epoch)

    def run_epoch(self, data, model, epoch, max_step=200):
        start = time.time()
        total_tokens = 0.
        total_loss = 0.
        tokens = 0.
        # print('data=', len(data), data)
        for i, batch in enumerate(data):
            if i>=max_step:
                break
            out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
            loss = self.compute_loss(out, batch.trg_y, batch.ntokens)

            total_loss += loss
            total_tokens += batch.ntokens
            tokens += batch.ntokens

            if i % 100 == 1:
                elapsed = time.time() - start
                print("Epoch %d Batch: %d Loss: %f Tokens per Sec: %fs" % (epoch, i - 1, loss / batch.ntokens, (tokens.float() / elapsed / 1000.)))
                start = time.time()
                tokens = 0

        return total_loss / total_tokens




    def evaluate(self, data):
        """
        在data上用训练好的模型进行预测，打印模型翻译结果
        """
        # 梯度清零
        with torch.no_grad():
            # 在data的英文数据长度上遍历下标
            for _ in range(10):
                i = random.randint(0, len(data.dev_en) - 1)
                # print('i=', i)
                # TODO: 打印待翻译的src句子
                cn_sent = " ".join([data.cn_index_dict[w] for w in data.dev_cn[i]])
                print("\n" + cn_sent)

                # TODO: 打印对应的句子答案
                en_sent = " ".join([data.en_index_dict[w] for w in data.dev_en[i]])
                print("".join(en_sent))

                # 将当前以单词id表示的英文句子数据转为tensor，并放如DEVICE中
                src = torch.from_numpy(np.array(data.dev_cn[i])).long().to(DEVICE)
                # 增加一维
                src = src.unsqueeze(0)
                # 设置attention mask
                src_mask = (src != 0).unsqueeze(-2)
                # 用训练好的模型进行decode预测
                out = self.greedy_decode(src, src_mask, max_len=self.MAX_LENGTH, start_symbol=data.en_word_dict["BOS"])
                # 初始化一个用于存放模型翻译结果句子单词的列表
                translation = []
                # 遍历翻译输出字符的下标（注意：开始符"BOS"的索引0不遍历）
                for j in range(1, out.size(1)):
                    # 获取当前下标的输出字符
                    sym = data.en_index_dict[out[0, j].item()]
                    # 如果输出字符不为'EOS'终止符，则添加到当前句子的翻译结果列表
                    if sym != 'EOS':
                        translation.append(sym)
                    # 否则终止遍历
                    else:
                        break
                # 打印模型翻译输出的中文句子结果
                print("translation: %s" % " ".join(translation))

    def greedy_decode(self, src, src_mask, max_len, start_symbol):
        """
        传入一个训练好的模型，对指定数据进行预测
        """
        # 先用encoder进行encode
        memory = self.model.encode(src, src_mask)
        # print('src=', src.shape, src)
        # print('src_mask=', src_mask.shape, src_mask)
        # print('memory=', memory.shape, memory)
        # 初始化预测内容为1×1的tensor，填入开始符('BOS')的id，并将type设置为输入数据类型(LongTensor)
        ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
        # print('ys=', ys.shape, ys)
        # 遍历输出的长度下标
        for i in range(max_len - 1):
            # decode得到隐层表示
            tgt_msk = subsequent_mask(ys.size(1)).type_as(src.data)
            # print('i=',i,'tgt_msk=', tgt_msk)
            out = self.model.decode(memory,
                               src_mask,
                               Variable(ys),
                               Variable(tgt_msk))

            # 获取当前位置最大概率的预测词id

            _, next_word = torch.max(out, dim=2)
            # print('i = ', i, ', out=', out.shape, out)
            # print('i = ', i, ', next_word=', next_word.shape, next_word)
            next_word = next_word.data[0, 0]
            # 将当前位置预测的字符id与之前的预测内容拼接起来
            ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
            # print('i = ', i, ' ,next_word=', next_word, ', ys=', ys)
        return ys

if __name__=='__main__':
    TRAIN_FILE = '../datas/eng-cmn-train.txt'
    DEV_FILE = '../datas/eng-cmn-eval.txt'
    TEST_FILE = ''
    SAVE_FILE = 'model.pt'
    h = 8
    d_model = 512
    d_ff = 2048
    dropout = 0.1
    N = 6
    epochs = 52
    batch_size = 64


    print('<<<<<<<<<<<<<<<start one step>>>>>>>>>>>>>>>>>>>>>\n1.start prepare data++++++++++++++++++++++\n')
    data = PrepareData(TRAIN_FILE, DEV_FILE, batch_size, DEVICE)
    print("src_vocab %d" % data.src_vocab)
    print("tgt_vocab %d" % data.tgt_vocab)
    print('self.train_en.len = ', len(data.train_en))
    print('self.dev_en.len = ', len(data.dev_en))

    print('\n2.start to train++++++++++++++++++++++++\n')
    transformer = Transformer(h, d_model, d_ff, dropout, N, data.src_vocab, data.tgt_vocab, DEVICE).to(DEVICE)
    trainer = Trainer(epochs, data.tgt_vocab, d_model, transformer, SAVE_FILE='model.pt', MAX_LENGTH=20)
    trainer.train(data)

    print('\n3.start to eval++++++++++++++++++++++++\n')
    transformer.load_state_dict(torch.load(SAVE_FILE))
    # trainer = Trainer(epochs, data.tgt_vocab, d_model, transformer, SAVE_FILE='model1.pt', MAX_LENGTH=5)
    trainer.evaluate(data)

