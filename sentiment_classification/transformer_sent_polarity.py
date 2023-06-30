"""import math

from torch.optim import optimizer
from torch import optim
import utils
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import tqdm


class PositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置编码
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # 不对位置编码层求梯度

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]  # 输入的词向量与位置编码相加
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class,
                 dropout=0.1, max_len=128, num_head=2, dim_feedforward=512,
                 activation='relu', num_layers=2):
        super(Transformer, self).__init__()
        # 词向量层
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        # 位置编码层
        self.position_embedding = PositionEncoding(embedding_dim, max_len)

        # 编码层：使用TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, num_head, dim_feedforward, dropout, activation)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # 输出层
        self.output = nn.Linear(hidden_dim, num_class)

    def forward(self, inputs, lengths):
        # 与lstm处理情况相同， 输入数据的第一维是批次，需要转换为TransformerEncoder所需要的第一维是长度，第二维是批次的形状
        inputs = torch.transpose(inputs, 0, 1)
        hidden_states = self.embeddings(inputs)
        hidden_states = self.position_embedding(hidden_states)
        attention_mask = utils.length_to_mask(lengths) == False
        # 根据批次中的每个序列长度生成Mask矩阵
        hidden_states = self.transformer(hidden_states, src_key_padding_mask=attention_mask)
        hidden_states = hidden_states[0, :, :]
        output = self.output(hidden_states)
        log_probs = F.log_softmax(output, dim=1)
        return log_probs


class TransformerDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def collate_fn(examples):
    lengths = torch.tensor([len(ex[0]) for ex in examples])
    inputs = [torch.tensor(ex[0]) for ex in examples]
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
    inputs = pad_sequence(inputs, batch_first=True)
    return inputs, lengths, targets


if __name__ == '__main__':
    embedding_dim = 128
    hidden_dim = 128
    num_class = 2
    batch_size = 32
    num_epoch = 100
    # 加载数据
    train_data, test_data, vocab = utils.load_sentence_polarity()
    train_dataset = TransformerDataset(train_data)
    test_dataset = TransformerDataset(test_data)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, collate_fn=collate_fn)

    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Transformer(len(vocab), embedding_dim, hidden_dim, num_class)
    model.to(device)

    # 训练过程
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器
    nll_loss = nn.NLLLoss()
    for epoch in range(num_epoch):
        total_loss = 0
        for batch in tqdm.tqdm(train_dataloader, desc=f'Trainng Epoch {epoch}'):
            inputs, lengths, targets = [x.to(device) for x in batch]
            log_probs = model(inputs, lengths)
            loss = nll_loss(log_probs, targets)
            optimizer.zero_grad()
            optimizer.step()
            total_loss += loss
        print(f'Loss: {total_loss:.2f}')

    # 测试过程
    model.eval()
    acc = 0
    for batch in tqdm.tqdm(test_dataloader, desc=f'Testing'):
        inputs, lengths, targets = [x.to(device) for x in batch]
        output = model(inputs, lengths)
        acc += (output.argmax(dim=1) == targets).sum().item()

    print(f"ACC:{acc / len(test_dataloader):.2f}")
"""

import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from collections import defaultdict
from vocab import Vocab
from utils import load_sentence_polarity, length_to_mask

# tqdm是一个Pyth模块，能以进度条的方式显式迭代的进度
from tqdm.auto import tqdm

class TransformerDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        return self.data[i]

def collate_fn(examples):
    lengths = torch.tensor([len(ex[0]) for ex in examples])
    inputs = [torch.tensor(ex[0]) for ex in examples]
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
    # 对batch内的样本进行padding，使其具有相同长度
    inputs = pad_sequence(inputs, batch_first=True)
    return inputs, lengths, targets

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # [max_len, 1] * [d_model / 2, ] --> [max_len, 1] * [1, d_model / 2]
        # --> [max_len, d_model/2] * [max_len, d_model/2] --> [max_len, d_model/2]
        pe[:, 0::2] = torch.sin(position * div_term)  # [512, 64]
        pe[:, 1::2] = torch.cos(position * div_term)  # [512, 64]
        pe = pe.unsqueeze(0).transpose(0, 1)  # [512, 128] --> [1   , 512, 128] --> [512, 1, 128]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
         位置编码考虑了所有的词，词向量与位置编码相加时需要对pe切片取前T个词
        :param x: 词向量[T, N, E]: [43, 32, 128]
        :return: 输入encoder的数据
        """
        # [512, 1, 128] -->
        x = x + self.pe[:x.size(0), :]
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class,
                 dim_feedforward=512, num_head=2, num_layers=2, dropout=0.1,
                 max_len=512, activation: str = "relu"):
        super(Transformer, self).__init__()
        # 词嵌入层
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = PositionalEncoding(embedding_dim, max_len)
        # 编码层：使用Transformer
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, num_head, dim_feedforward, dropout, activation)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        # 输出层
        self.output = nn.Linear(hidden_dim, num_class)

    def forward(self, inputs, lengths):
        inputs = torch.transpose(inputs, 0, 1)
        hidden_states = self.embeddings(inputs)
        hidden_states = self.position_embedding(hidden_states)
        attention_mask = length_to_mask(lengths) == False
        hidden_states = self.transformer(hidden_states, src_key_padding_mask=attention_mask)
        hidden_states = hidden_states[0, :, :]
        output = self.output(hidden_states)
        log_probs = F.log_softmax(output, dim=1)
        return log_probs

embedding_dim = 128
hidden_dim = 128
num_class = 2
batch_size = 32
num_epoch = 20

# 加载数据
train_data, test_data, vocab = load_sentence_polarity()
train_dataset = TransformerDataset(train_data)
test_dataset = TransformerDataset(test_data)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Transformer(len(vocab), embedding_dim, hidden_dim, num_class)
model.to(device) # 将模型加载到GPU中（如果已经正确安装）

# 训练过程
nll_loss = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) # 使用Adam优化器

model.train()

for epoch in range(num_epoch):
    total_loss = 0
    acc = 0
    count = 0
    for batch in tqdm(train_data_loader, desc=f"Training Epoch {epoch}"):
        inputs, lengths, targets = [x.to(device) for x in batch]
        log_probs = model(inputs, lengths)
        loss = nll_loss(log_probs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        acc += (log_probs.argmax(dim=1) == targets).sum().item()
        count += batch_size
    print(f"Loss: {total_loss:.3f}")
    print(f'Acc:{acc / count:.3f}')

# 测试过程
acc = 0
for batch in tqdm(test_data_loader, desc=f"Testing"):
    inputs, lengths, targets = [x.to(device) for x in batch]
    with torch.no_grad():
        output = model(inputs, lengths)
        acc += (output.argmax(dim=1) == targets).sum().item()

# 输出在测试集上的准确率
print(f"Acc: {acc / len(test_data_loader):.3f}")