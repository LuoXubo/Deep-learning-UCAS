"""
@Description :   
@Author      :   Xubo Luo 
@Time        :   2024/06/10 20:38:19
"""

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
# from torchnet import meter
import random
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepareData():
    datas = np.load("../../Dataset/tang.npz", allow_pickle=True)
    data = datas['data']
    ix2word = datas['ix2word'].item()
    word2ix = datas['word2ix'].item()
    data = preprocess(data)
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    return dataloader, ix2word, word2ix

def preprocess(sentences):
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

class PoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(PoetryModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = LSTM_LAYER
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=self.num_layers)
        self.linear = nn.Linear(self.hidden_dim, vocab_size)
        self.fc = nn.Sequential(nn.Linear(self.hidden_dim, 2048),
                                nn.Tanh(),
                                nn.Linear(2048, vocab_size))
        self.dropout = nn.Dropout(0.6)

    def forward(self, input, hidden=None):
        seq_len, batch_size = input.size()
        if hidden is None:
            h_0 = input.data.new(self.num_layers, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(self.num_layers, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        embeds = self.embeddings(input)
        output, hidden = self.lstm(embeds, (h_0, c_0))
        output = self.dropout(output)
        output = self.fc(output.view(seq_len * batch_size, -1))
        return output, hidden

class PoetryModel2(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(PoetryModel2, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = LSTM_LAYER
        self.lstm_outdim = LSTM_OUTDIM
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=self.num_layers)
        self.lstm2 = nn.LSTM(self.hidden_dim, self.lstm_outdim, num_layers=self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, vocab_size)
        self.fc = nn.Sequential(nn.Linear(self.lstm_outdim, 2048),
                                nn.Tanh(),
                                nn.Linear(2048, vocab_size))
        self.dropout = nn.Dropout(0.6)

    def forward(self, input, hidden1=None, hidden2=None):
        seq_len, batch_size = input.size()
        if hidden1 is None or hidden2 is None:
            h_0 = input.data.new(self.num_layers, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(self.num_layers, batch_size, self.hidden_dim).fill_(0).float()
            h_1 = input.data.new(self.num_layers, seq_len, self.lstm_outdim).fill_(0).float()
            c_1 = input.data.new(self.num_layers, seq_len, self.lstm_outdim).fill_(0).float()
        else:
            h_0, c_0 = hidden1
            h_1, c_1 = hidden2
        embeds = self.embeddings(input)
        output, hidden1 = self.lstm1(embeds, (h_0, c_0))
        output, hidden2 = self.lstm2(output, (h_1, c_1))
        output = self.dropout(output)
        output = self.fc(output.reshape(seq_len * batch_size, -1))
        return output, hidden1, hidden2

def train(epochs, poem_loader, word2ix):
    model = PoetryModel2(len(word2ix), embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM)
    model.train()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
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

        torch.save(model.state_dict(),"caches/tang{}_.pth".format(epoch))

def generate(model, start_words, ix2word, word2ix):
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
    return results

def gen_acrostic(model, start_words, ix2word, word2ix):
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
    return result

if __name__ == '__main__':
    poem_loader, ix2word, word2ix = prepareData()
    # train(EPOCHS, poem_loader, word2ix)
    # 读取模型
    model = PoetryModel2(len(word2ix), EMBEDDING_DIM, HIDDEN_DIM,)
    model.load_state_dict(torch.load("caches/tang19_.pth",map_location=torch.device('cpu')))
    results1 = generate(model, '花开花落几番时', ix2word, word2ix)
    results2 = gen_acrostic(model, '深度学习', ix2word, word2ix)
    print(' '.join(i for i in results1))
    print(' '.join(i for i in results2))