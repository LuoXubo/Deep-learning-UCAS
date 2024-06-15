"""
@Description :   
@Author      :   Xubo Luo 
@Time        :   2024/06/15 15:30:46
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class Encoder(BaseModel):

    def __init__ (self, vocab_size, h_dim, pf_dim, n_heads, n_layers, dropout, device, max_seq len=200):
        super().__init__()
        self.n_layers = n_layers
        self.h_dim = h_dim
        self.device = device
        self.word_embeddings = WordEmbeddings(vocab_size, h_dim)
        self.pe = PositionEmbeddings(max_seq_len, h_dim)
        self.layers = nn.ModuleList()

        for i in range(n_layers):
            self.layers.append(EncoderLayer(h_dim, n_heads, pf_dim, dropout, device))

        self.dropout = nn.Dropout (dropout)
        self.scale = torch.sqgrt(torch.FloatTensor([h_dim])).to(device)

        def forward(self, src, src_mask):
            output = self.word_embeddings(src) * self.scale
            src_len = src.shape[1]
            pos = torch.arange(0, src_len).unsqueeze().repeat(src.shape[0], 1).to(self.device)
            output = self.dropout(output + self.pe(pos))
            # output = self.pe (output)

            for i in range(self.n_layers):
                output = self.layers[i](output, src_mask)

            return output