"""
@Description :   
@Author      :   Xubo Luo 
@Time        :   2024/06/10 20:29:52
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math

# 加载数据集
data = np.load('../../Dataset/tang.npz', allow_pickle=True)
ix2word = data['ix2word'].tolist()
word2ix = data['word2ix'].tolist()
poetrys = data['data']

# 超参数设置
vocab_size = len(ix2word)
embedding_dim = 256
hidden_dim = 512
num_heads = 8
num_layers = 6
max_len = 125  # 根据数据集中的诗句长度决定

# 定义位置编码类
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

# 定义Transformer模型
class PoetryTransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_heads, num_layers):
        super(PoetryTransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim)
        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim
        )
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        src_emb = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        tgt_emb = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        src_emb = self.positional_encoding(src_emb)
        tgt_emb = self.positional_encoding(tgt_emb)

        output = self.transformer(
            src_emb, tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask
        )

        output = self.linear(output)
        return output

# 准备数据
def prepare_data(poetrys, word2ix, max_len):
    data = []
    for poetry in poetrys:
        indices = [word2ix[word] for word in poetry]
        if len(indices) < max_len:
            indices += [word2ix['<PAD>']] * (max_len - len(indices))
        data.append(indices)
    return torch.tensor(data)

# 数据加载
data_tensor = prepare_data(poetrys, word2ix, max_len)
data_loader = torch.utils.data.DataLoader(data_tensor, batch_size=32, shuffle=True)

# 实例化模型
model = PoetryTransformerModel(vocab_size, embedding_dim, hidden_dim, num_heads, num_layers)
criterion = nn.CrossEntropyLoss(ignore_index=word2ix['<PAD>'])
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in data_loader:
        src = batch[:, :-1]
        tgt = batch[:, 1:]
        src_mask = model.transformer.generate_square_subsequent_mask(src.size(1))
        tgt_mask = model.transformer.generate_square_subsequent_mask(tgt.size(1))

        optimizer.zero_grad()
        output = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        loss = criterion(output.view(-1, vocab_size), tgt.contiguous().view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_loss:.4f}')

# 保存模型
torch.save(model.state_dict(), 'transformer.pth')

# 测试模型
model.eval()
start_words = '春江花月夜'
start_indices = [word2ix[word] for word in start_words]
start_tensor = torch.tensor(start_indices).unsqueeze(0)
start_tensor = torch.cat([start_tensor, torch.zeros(1, max_len-len(start_indices), dtype=torch.long)], dim=1)
output = start_tensor
for i in range(max_len-len(start_indices)):
    src_mask = model.transformer.generate_square_subsequent_mask(output.size(1))
    tgt_mask = model.transformer.generate_square_subsequent_mask(output.size(1))
    output = model(output, output, src_mask=src_mask, tgt_mask=tgt_mask)
    output = torch.cat([start_tensor[:, :i+1], output[:, i+1:]], dim=1)
    output_ids = output.argmax(dim=-1)
    if output_ids[0, i] == word2ix['<EOS>']:
        break

output_ids = output_ids[0].numpy().tolist()
output_words = [ix2word[ix] for ix in output_ids]
print(''.join(output_words))
