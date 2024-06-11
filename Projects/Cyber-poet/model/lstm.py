"""
@Description :   The model for the project
@Author      :   Xubo Luo 
@Time        :   2024/05/24 17:34:50
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTM, self).__init__()
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


class DoubleLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(DoubleLSTM, self).__init__()
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
    
if __name__ == '__main__':
    model = DoubleLSTM(150, EMBEDDING_DIM, HIDDEN_DIM,)
    input = torch.randint(0, 150, (10, 32))
    output, hidden1, hidden2 = model(input)
    print(output.shape, hidden1[0].shape, hidden1[1].shape, hidden2[0].shape, hidden2[1].shape)





