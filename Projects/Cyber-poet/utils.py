"""
@Description :   The utils for the project
@Author      :   Xubo Luo 
@Time        :   2024/05/24 17:34:38
"""

import os
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
import random
import tqdm


random.seed(20)
torch.manual_seed(20)
EMBEDDING_DIM = 512
HIDDEN_DIM = 1024
LSTM_OUTDIM = 512
LR = 0.001
MAX_GEN_LEN = 200
EPOCHS = 20
DROP_PROB = 0.5
LSTM_LAYER = 3
BATCH_SIZE = 16


def get_data(opt):
    datas = np.load(os.path.join(opt.data_path, opt.filename), allow_pickle=True)
    data, word2ix, ix2word = datas['data'], datas['word2ix'].item(), datas['ix2word'].item()
    return data, word2ix, ix2word


def trans_ix2word(data, ix2word):
    words = []
    for i in data:
        words.append(ix2word[i])

    return words


def fPrint(wordList):
    for i in wordList:
        if i == '，':
            print(i, end='\t')
        elif i in ['。', '？', '！']:
            print(i, end='\n')
        else:
            print(i, end='')


def prepareData(PATH, BATCH_SIZE):
    """
    :param PATH: the path of the data
    :param BATCH_SIZE: the batch size
    :return: the dataloader, the ix2word and the word2ix
    """
    datas = np.load(PATH, allow_pickle=True)
    data = datas['data']
    ix2word = datas['ix2word'].item()
    word2ix = datas['word2ix'].item()
    data = preprocess(data)
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    return dataloader, ix2word, word2ix

def preprocess(sentences):
    """
    :param sentences: the sentences
    :return: the preprocessed sentences
    """
    new_sentences = []
    for sentence in sentences:
        new_sentence = [token for token in sentence if token != 8292]
        if len(new_sentence) < 125:
            new_sentence.extend([8292] * (125 - len(new_sentence)))
        else:
            new_sentence = new_sentence[:125]
        new_sentences.append(new_sentence)
    sentences = np.array(new_sentences)
    sentences = torch.tensor(sentences, dtype=torch.long)
    return sentences

def generate(model, start_words, ix2word, word2ix, device):
    """
    :param model: the model
    :param start_words: the start words
    :param ix2word: the ix2word
    :param word2ix: the word2ix
    :param device: the device
    :return: the generated words
    """
    model.eval()
    results = list(start_words)
    start_word_len = len(start_words)
    input = torch.Tensor([word2ix['<START>']]).view(1, 1).long()
    input = input.to(device)
    h1, h2 = None, None
    model = model.to(device)
    model.eval()

    for i in range(50):
        output, h1, h2 = model(input, h1, h2)
        if i < start_word_len:
            w = results[i]
            input = input.data.new([word2ix[w]]).view(1, 1)
        else:
            top_index = output.data[0].topk(1)[1][0].item()
            w = ix2word[top_index]
            results.append(w)
            input = input.data.new([top_index]).view(1, 1)
        if w == '<EOP>':
            del results[-1]
            break
    post_res = ' '.join(i for i in results)

    return post_res

def gen_acrostic(model, start_words, ix2word, word2ix, device):
    """
    :param model: the model
    :param start_words: the start words
    :param ix2word: the ix2word
    :param word2ix: the word2ix
    :param device: the device
    :return: the generated words
    """
    result = []
    start_words_len = len(start_words)
    input = torch.Tensor([word2ix['<START>']]).view(1, 1).long()
    index = 0
    pre_word = '<START>'
    h1, h2 = None, None
    model = model.to(device)
    model.eval()
    input = input.to(device)

    for i in range(125):
        output, h1, h2 = model(input, h1, h2)
        top_index = output.data[0].topk(1)[1][0].item()
        w = ix2word[top_index]
        if pre_word in {'。', '，', '?', '！', '<START>'}:
            if index == start_words_len:
                break
            else:
                w = start_words[index]
                index += 1
                input = (input.data.new([word2ix[w]])).view(1, 1)
        else:
            input = (input.data.new([top_index])).view(1, 1)
        result.append(w)
        pre_word = w
    
    post_res = ' '.join(i for i in result)

    return post_res

def train(model_name, model, epochs, poem_loader, word2ix, device):
    """
    :param model_name: the model name
    :param model: the model
    :param epochs: the epochs
    :param poem_loader: the dataloader
    :param word2ix: the word2ix
    :param device: the device
    :return: None
    """
    model.train()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    for epoch in tqdm.tqdm(range(epochs)):
        for batch_idx, data in enumerate(poem_loader):
            data = data.long().transpose(1,0).contiguous()
            data = data.to(device)
            optimizer.zero_grad()
            input, target = data[:-1, :], data[1:, :]
            output, _, _ = model(input)
            loss = criterion(output, target.view(-1))
            loss.backward()
            optimizer.step()
            if batch_idx % 900 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch + 1, batch_idx * len(data[1]), len(poem_loader.dataset),
                    100. * batch_idx / len(poem_loader), loss.item()))

        if epoch % 5 == 0 or epoch == epochs - 1:
            torch.save(model.state_dict(),"caches/{}_{}.pth".format(model_name, epoch))